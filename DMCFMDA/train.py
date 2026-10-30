"""DMCFMDA's native training entry, adapted only for the shared CV protocol."""

import json
import random
import argparse
from pathlib import Path
import sys

_RUNTIME = Path(__file__).resolve().parent / ".runtime"
if _RUNTIME.exists():
    sys.path.insert(0, str(_RUNTIME))

import dgl
import networkx as nx
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch import optim
from torch_geometric.utils import dense_to_sparse
from tqdm import trange

from DMCFMDA import DMCFMDA
from utils import k_matrix


# The repository's supported Windows DGL wheel is CPU-only.  Keep every DMC
# component on the same device; CUDA is still used by the other baselines.
device = th.device("cpu")
BALANCED = Path(__file__).resolve().parents[1] / "peryton_global_positive_balanced5fold_v1"


def loss_contrastive_m(m1, m2):
    m1, m2 = m1 / th.norm(m1), m2 / th.norm(m2)
    pos = th.sum(m1 * m2, dim=1, keepdim=True)
    neg1 = m1 @ m1.t() - th.diag_embed(th.diag(m1 @ m1.t()))
    neg2 = m2 @ m2.t() - th.diag_embed(th.diag(m2 @ m2.t()))
    return th.mean(F.softplus(th.mean(th.cat([neg1, neg2], dim=1), dim=1) - pos.squeeze(1)))


def loss_contrastive_d(d1, d2):
    d1, d2 = d1 / th.norm(d1), d2 / th.norm(d2)
    pos = th.sum(d1 * d2, dim=1, keepdim=True)
    neg1 = d1 @ d1.t() - th.diag_embed(th.diag(d1 @ d1.t()))
    neg2 = d2 @ d2.t() - th.diag_embed(th.diag(d2 @ d2.t()))
    return th.mean(F.softplus(th.mean(th.cat([neg1, neg2], dim=1), dim=1) - pos.squeeze(1)))


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def _dgl_graph(matrix):
    return dgl.from_networkx(nx.from_numpy_array((matrix > 0).astype(np.float32)))


def _edge_index(matrix):
    return dense_to_sparse(th.as_tensor(matrix, dtype=th.float32))[0]


def _fixed_association_graph(edges, microbe_count, disease_count):
    edges = edges.copy()
    edges[:, 1] += microbe_count
    source = np.concatenate((edges[:, 0], edges[:, 1]))
    target = np.concatenate((edges[:, 1], edges[:, 0]))
    return (
        dgl.graph((source, target), num_nodes=microbe_count + disease_count),
        th.as_tensor(np.vstack((source, target)), dtype=th.long),
    )


def _write_results(directory, metrics, rocs, prs):
    directory.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 1001)
    np.savetxt(directory / "fpr.txt", np.tile(grid[:, None], (1, 5)))
    np.savetxt(directory / "tpr.txt", np.column_stack([np.interp(grid, x, y) for x, y in rocs]))
    np.savetxt(directory / "recall.txt", np.tile(grid[:, None], (1, 5)))
    np.savetxt(directory / "precision.txt", np.column_stack([np.interp(grid, x, y) for x, y in prs]))
    columns = ["AUC", "AUPR", "ACC", "Sensitivity", "Specificity", "Precision"]
    values = np.asarray([[entry[key] for key in columns] for entry in metrics])
    with open(directory / "metrics.txt", "w", encoding="utf-8") as output:
        output.write("fold\t" + "\t".join(columns) + "\n")
        for fold, row in enumerate(values, 1):
            output.write(f"{fold}\t" + "\t".join(f"{x:.8f}" for x in row) + "\n")
        output.write("mean\t" + "\t".join(f"{x:.8f}" for x in values.mean(0)) + "\n")
        output.write("std\t" + "\t".join(f"{x:.8f}" for x in values.std(0, ddof=1)) + "\n")
    with open(directory / "protocol.json", "w", encoding="utf-8") as output:
        json.dump({"split": "global positive-edge 5-fold", "negative_sampling": "disabled",
                   "loss_class_balance": "equal mean BCE of all training positives and all unknown pairs",
                   "topology": "fixed full association topology", "auc_aupr": "continuous scores"}, output, indent=2)


def train(data, args):
    """Train the original DMCFMDA model on all candidates in each fixed fold."""
    args.device = str(device)
    labels = data["labels"]
    mm = k_matrix(data["ms"], args.neighbor)
    dd = k_matrix(data["ds"], args.neighbor)
    mm_graph, dd_graph = _dgl_graph(mm), _dgl_graph(dd)
    mm_edges, dd_edges = _edge_index(mm), _edge_index(dd)
    md_graph, md_edges = _fixed_association_graph(data["topology_md"], args.microbe, args.disease_number)
    microbe = th.as_tensor(data["ms"], dtype=th.float32, device=device)
    disease = th.as_tensor(data["ds"], dtype=th.float32, device=device)
    all_metrics, rocs, prs = [], [], []

    for fold in range(5):
        _seed(args.random_seed + fold)
        held = data["fold_assignment"] == fold
        if np.any(labels[held] != 1) or np.any(data["fold_assignment"][labels == 0] != -1):
            raise RuntimeError("Only positive associations may be assigned to a cross-validation fold.")
        fold_dir = BALANCED / f"fold_{fold + 1}"
        # Manifest is disease,microbe; DMCFMDA's native samples are microbe,disease.
        train_pos = np.loadtxt(fold_dir / "train_positive_disease_microbe.txt", dtype=np.int64).reshape(-1, 2)[:, [1, 0]]
        train_neg = np.loadtxt(fold_dir / "train_negative_disease_microbe.txt", dtype=np.int64).reshape(-1, 2)[:, [1, 0]]
        eval_pos = np.loadtxt(fold_dir / "test_positive_disease_microbe.txt", dtype=np.int64).reshape(-1, 2)[:, [1, 0]]
        eval_neg = np.loadtxt(fold_dir / "test_negative_disease_microbe.txt", dtype=np.int64).reshape(-1, 2)[:, [1, 0]]
        train_array = np.vstack((np.c_[train_pos, np.ones(len(train_pos), dtype=np.int64)],
                                 np.c_[train_neg, np.zeros(len(train_neg), dtype=np.int64)]))
        eval_array = np.vstack((np.c_[eval_pos, np.ones(len(eval_pos), dtype=np.int64)],
                                np.c_[eval_neg, np.zeros(len(eval_neg), dtype=np.int64)]))
        eval_labels = eval_array[:, 2]
        train_tensor = th.as_tensor(train_array, dtype=th.long, device=device)
        eval_tensor = th.as_tensor(eval_array, dtype=th.long, device=device)
        model = DMCFMDA(args).to(device)
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
        print(f"################Fold {fold + 1} of 5################")
        print(
            f"protocol audit: held_positive={len(eval_pos)}, "
            f"train_positive={train_array[:, 2].sum()}, "
            f"train_negative={len(train_neg)}, eval_candidates={len(eval_array)}"
        )
        for epoch in trange(args.epochs, desc=f"DMCFMDA fold {fold + 1}"):
            model.train()
            optimizer.zero_grad()
            scores, m1, m2, d1, d2 = model(
                mm_graph, dd_graph, md_graph, microbe, disease, train_tensor, epoch, Issne=True,
                mm_edge_index=mm_edges, dd_edge_index=dd_edges, md_edge_index=md_edges,
            )
            # The balanced manifest supplies the same positive/negative count
            # as the original DMCFMDA training objective.
            target = train_tensor[:, 2].float()
            prediction = scores.flatten()
            positive = target == 1
            supervised_loss = 0.5 * (
                F.binary_cross_entropy(prediction[positive], target[positive])
                + F.binary_cross_entropy(prediction[~positive], target[~positive])
            )
            loss = supervised_loss + loss_contrastive_m(m1, m2) + loss_contrastive_d(d1, d2)
            loss.backward()
            optimizer.step()
            if epoch == args.epochs - 1 or (epoch + 1) % 50 == 0:
                print(f"fold={fold + 1}, epoch={epoch + 1}, loss={loss.item():.6f}")
        model.eval()
        with th.no_grad():
            scores, *_ = model(
                mm_graph, dd_graph, md_graph, microbe, disease, eval_tensor, args.epochs, Issne=False,
                mm_edge_index=mm_edges, dd_edge_index=dd_edges, md_edge_index=md_edges,
            )
        scores = scores.flatten().detach().cpu().numpy()
        fpr, tpr, threshold_values = roc_curve(eval_labels, scores)
        threshold = threshold_values[np.argmax(tpr - fpr)]
        predicted = (scores >= threshold).astype(int)
        tn = np.sum((eval_labels == 0) & (predicted == 0))
        fp = np.sum((eval_labels == 0) & (predicted == 1))
        precision, recall, _ = precision_recall_curve(eval_labels, scores)
        metric = {"AUC": roc_auc_score(eval_labels, scores),
                  "AUPR": average_precision_score(eval_labels, scores),
                  "ACC": accuracy_score(eval_labels, predicted),
                  "Sensitivity": recall_score(eval_labels, predicted, zero_division=0),
                  "Specificity": tn / (tn + fp),
                  "Precision": precision_score(eval_labels, predicted, zero_division=0)}
        all_metrics.append(metric)
        rocs.append((fpr, tpr))
        prs.append((recall[::-1], precision[::-1]))
        print(f"fold {fold + 1}: {metric}")
    _write_results(Path(args.result_dir) / "train_global5fold_balanced_samples", all_metrics, rocs, prs)
    return all_metrics


def main():
    """Native command-line entry point for DMCFMDA training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Peryton")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_feats", type=int, default=64)
    parser.add_argument("--out_feats", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp", type=int, nargs="+", default=[64, 1])
    parser.add_argument("--neighbor", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--gt_layer", type=int, default=2)
    parser.add_argument("--gt_head", type=int, default=1)
    args = parser.parse_args()
    args.data_dir = f"data/{args.dataset}/"
    args.result_dir = f"result/{args.dataset}/"
    from utils import data_processing, get_data
    data = get_data(args)
    args.microbe = data["microbe"]
    args.disease_number = data["disease_number"]
    data_processing(data, args)
    train(data, args)


if __name__ == "__main__":
    main()
