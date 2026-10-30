"""Run DMCFMDA on the shared fixed multiMDA five-fold split."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

BASELINE_DIR = Path(__file__).resolve().parents[1]
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from multiMDA_python_common import evaluate_and_save, load_fixed_multimda  # noqa: E402
from DMCFMDA import DMCFMDA  # noqa: E402
from utils import k_matrix  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dgl_graph(matrix, device):
    graph_nx = nx.from_numpy_array((matrix > 0).astype(np.float32))
    return dgl.from_networkx(graph_nx).to(device)


def make_edge_index(matrix, device):
    return dense_to_sparse(torch.as_tensor(matrix, dtype=torch.float32, device=device))[0]


def make_args(num_microbes, num_diseases, epochs, lr, dropout, device):
    class Args:
        pass

    args = Args()
    args.microbe = num_microbes
    args.disease_number = num_diseases
    args.in_feats = 64
    args.out_feats = 128
    args.gt_layer = 2
    args.gt_head = 1
    args.dropout = dropout
    args.mlp = [64, 1]
    args.epochs = epochs
    args.lr = lr
    args.wd = 1e-3
    args.device = str(device)
    return args


def run_fold(fixed_data, fold_id, epochs, seed, device, negative_rate):
    set_seed(seed + fold_id)
    labels = fixed_data["full_labels"]
    train_matrix = fixed_data["train_matrices"][fold_id]
    num_diseases, num_microbes = labels.shape
    args = make_args(num_microbes, num_diseases, epochs, 1e-3, 0.2, device)

    microbe_features_np = fixed_data["microbe_features"].astype(np.float32)
    disease_features_np = fixed_data["disease_features"].astype(np.float32)
    microbe_features = torch.from_numpy(microbe_features_np).to(device)
    disease_features = torch.from_numpy(disease_features_np).to(device)
    mm_matrix = k_matrix(microbe_features_np, 30)
    dd_matrix = k_matrix(disease_features_np, 30)
    mm_dgl = make_dgl_graph(mm_matrix, device)
    dd_dgl = make_dgl_graph(dd_matrix, device)
    mm_edges = make_edge_index(mm_matrix, device)
    dd_edges = make_edge_index(dd_matrix, device)

    train_pos = np.argwhere(train_matrix == 1)
    all_neg = np.argwhere(labels == 0)
    rng = np.random.default_rng(seed + fold_id)
    n_neg = min(len(all_neg), int(round(negative_rate * len(train_pos))))
    neg_pos = rng.choice(len(all_neg), size=n_neg, replace=False)
    train_neg = all_neg[neg_pos]
    train_samples = np.vstack([
        np.column_stack([train_pos[:, 1], train_pos[:, 0], np.ones(len(train_pos))]),
        np.column_stack([train_neg[:, 1], train_neg[:, 0], np.zeros(len(train_neg))]),
    ]).astype(np.int64)
    train_samples = torch.from_numpy(train_samples).to(device)

    md_edges = np.vstack([train_pos[:, 1], train_pos[:, 0] + num_microbes])
    md_edge_index = torch.from_numpy(md_edges.astype(np.int64)).to(device)
    model = DMCFMDA(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred, m1, m2, d1, d2 = model(
            mm_dgl, dd_dgl, md_edge_index,
            microbe_features, disease_features, train_samples, epoch,
            Issne=True, mm_edge_index=mm_edges, dd_edge_index=dd_edges,
            md_edge_index=md_edge_index,
        )
        target = train_samples[:, 2].float()
        loss = F.binary_cross_entropy(pred.reshape(-1), target)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            print(f"DMCFMDA Fold {fold_id + 1}: epoch {epoch + 1}/{epochs}, loss={loss.item():.6f}")

    # Predict every disease-microbe pair for GECLMDA-compatible full-matrix
    # ROC/PR curves.  The evaluator derives the masked train==0 metrics from
    # this same complete score matrix as a secondary report.
    eval_pairs = np.argwhere(np.ones_like(labels, dtype=bool))
    score_matrix = np.zeros_like(labels, dtype=np.float32)
    batch_size = 32768
    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_pairs), batch_size):
            batch_pairs = eval_pairs[start : start + batch_size]
            samples = torch.from_numpy(
                np.column_stack([batch_pairs[:, 1], batch_pairs[:, 0], np.zeros(len(batch_pairs))]).astype(np.int64)
            ).to(device)
            pred, *_ = model(
                mm_dgl, dd_dgl, md_edge_index,
                microbe_features, disease_features, samples, epochs,
                Issne=False, mm_edge_index=mm_edges, dd_edge_index=dd_edges,
                md_edge_index=md_edge_index,
            )
            score_matrix[batch_pairs[:, 0], batch_pairs[:, 1]] = pred.reshape(-1).cpu().numpy()
    return score_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--negative_rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    fixed_data = load_fixed_multimda(args.fixed_data_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(args.device)
    print(f"DMCFMDA device: {device}")
    scores = [
        run_fold(fixed_data, fold_id, args.epochs, args.seed, device, args.negative_rate)
        for fold_id in range(5)
    ]
    summary = evaluate_and_save("DMCFMDA", scores, fixed_data)
    print(f"DMCFMDA mean AUC={summary['AUC_mean']:.6f}, mean AUPR={summary['AUPR_mean']:.6f}")


if __name__ == "__main__":
    main()
