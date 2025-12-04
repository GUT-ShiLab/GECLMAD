import datetime
import os
import pickle
import re
import time
import argparse
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model.model import MDA_Graph
from src.utils.utils import specificity
from utils.utils import accuracy, mcc, auc, aupr, f1, precision, recall, find_optimal_cutoff
import numpy as np
import pandas as pd
import torch
from utils.load_data import load_data
from sklearn import metrics

from src.data.process_data import cross_5_folds, first_spilt_label
import warnings

project_root = Path(__file__).parent.parent

warnings.filterwarnings('ignore')

# 123456789123456
###############################################################
# Training settings
parser = argparse.ArgumentParser(description='MDA-GRAPH')
parser.add_argument('--seed', type=int, default=1145141919, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2500,  # 10000
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,  # 0.0005
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.2)

###############################################################
# Model hyper setting
# microbe_NN
parser.add_argument('--microbe_ninput', type=int, default=256,  # 64  128 512
                    help='microbe vector size')
parser.add_argument('--nn_nlayers', type=int, default=1,
                    help='microbe_nn layers num')
parser.add_argument('--pnn_nhid', type=str, default='[]',
                    help='pnn hidden layer dim,  like [200,100] for tow hidden layers')
parser.add_argument('--dnn_nlayers', type=int, default=1,
                    help='dnn_nlayers num')
parser.add_argument('--dnn_nhid', type=str, default='[]',
                    help='dnn hidden layer dim, like [200,100] for tow hidden layers')
# Decoder
parser.add_argument('--DTI_nn_nlayers', type=int, default=1,
                    help='microbe_nn layers num')
parser.add_argument('--DTI_nn_nhid', type=str, default='[128,128,128]',
                    help='DTI_nn hidden layer dim, like [200,100] for tow hidden layers')
###############################################################
# data
parser.add_argument('--dataset', type=str, default='HMDAD',  # disbiome
                    help='dataset name')
parser.add_argument('--data_path', type=str, default=f'{project_root}/data/raw/',
                    help='dataset root path')
parser.add_argument('--pos_num', type=int, default=10,  # 2,4...10
                    help='pos_num, the pos sample number for each node in contrast')

parser.add_argument('--key', type=bool, default=1, help='get the data,set 1,or 0')
parser.add_argument('--savedata', type=bool, default=1, help='save the data,set 1,or 0')
###############################################################
# Encoder
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--tau', type=float, default=0.8)
parser.add_argument('--feat_drop', type=float, default=0.1)
parser.add_argument('--attn_drop', type=float, default=0.1)
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--mp_ngcn', type=int, default=1)
parser.add_argument('--sn_ngcn', type=int, default=1)
parser.add_argument("--encoder_loss_rate", type=float, default=0.5)
parser.add_argument('--embs_rate', type=float, default=1)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
np.random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.manual_seed(args.seed)

# nn layers
p1 = re.compile(r'[[](.*?)[]]', re.S)
if args.dnn_nhid == '[]':
    args.dnn_nhid = []
else:
    args.dnn_nhid = [int(i) for i in re.findall(p1, args.dnn_nhid)[0].replace(' ', '').split(',')]
if args.pnn_nhid == '[]':
    args.pnn_nhid = []
else:
    args.pnn_nhid = [int(i) for i in re.findall(p1, args.pnn_nhid)[0].replace(' ', '').split(',')]
args.DTI_nn_nhid = [int(i) for i in re.findall(p1, args.DTI_nn_nhid)[0].replace(' ', '').split(',')]

# Hyper Setting
pnn_hyper = [args.microbe_ninput, args.pnn_nhid, args.hidden_dim, args.nn_nlayers]
Deco_hyper = [args.hidden_dim, args.DTI_nn_nhid, args.DTI_nn_nlayers]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, posw=24):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=posw, reduction='none')

    def forward(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets)
        probs = torch.sigmoid(outputs)
        alpha = self.alpha
        gamma = self.gamma

        # 计算 Focal Loss
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_loss = alpha * ((1 - p_t) ** gamma) * bce_loss
        return focal_loss.mean()


def train(epoch, link_dti_id_train, train_dti_inter_mat, feats, pos, mps, nei_index, sn_edge, mp_edge):
    """
    link_dti_id_train: training set of disease-microbe interaction pairs, pos and neg
    train_dti_inter_mat: disease-microbe interaction mat, 1 for pos, 0 for neg used in test, -1 for other negs
    feats: input embeddings
    pos: pos pairs for contrast
    mps: meta-path network constructed previously
    nei_index: normalized sim adj matrix (DAD) for GCN
    sn_edge: similarity network edges
    mp_edge: meta-path network edges
    """

    t = time.time()
    model.train()
    optimizer.zero_grad()
    row_dti_id = link_dti_id_train.permute(1, 0)[0]
    col_dti_id = link_dti_id_train.permute(1, 0)[1]

    disease_index = row_dti_id
    microbe_index = col_dti_id + train_dti_inter_mat.shape[0]
    ADJ = train_dti_inter_mat[row_dti_id, col_dti_id]
    preds, loss_encoder, encoder_embeds = model(microbe_index, disease_index, feats, pos, mps, nei_index, sn_edge,
                                                mp_edge, ADJ)

    # _,threshold_value = find_optimal_cutoff(preds, train_dti_inter_mat[row_dti_id, col_dti_id])
    # pd = preds.ge(threshold_value)

    # Loss = nn.BCELoss()
    # BCEnn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    num_pos = train_dti_inter_mat[row_dti_id, col_dti_id].sum()
    # print('disease_features',train_dti_inter_mat[row_dti_id, col_dti_id].shape)
    num_neg = len(preds) - num_pos
    pw = num_neg / num_pos
    # # # weight=torch.tensor([pw]).cuda()
    # # # pos_weight=weight
    # Loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).cuda())
    # loss_train = Loss(preds, train_dti_inter_mat[row_dti_id, col_dti_id])
    acc_dti_train = accuracy(encoder_embeds, train_dti_inter_mat[row_dti_id, col_dti_id])
    focal_loss_fn = FocalLoss(alpha=pw * 2, gamma=2, posw=pw)
    loss_train = focal_loss_fn(preds, train_dti_inter_mat[row_dti_id, col_dti_id])

    loss_all = loss_train + loss_encoder
    loss_all.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
        print('Epoch {:04d} Train '.format(epoch + 1),
              'loss_all: {:.4f}'.format(loss_all.item()),
              'acc_dti_train: {:.4f}'.format(acc_dti_train),
              'time: {:.4f}s'.format(time.time() - t))
    optimizer.zero_grad()


def test(link_dti_id_test, test_dti_inter_mat, feats, pos, mps, nei_index, sn_edge, mp_edge):
    """
    link_dti_id_test: test set of disease-microbe interaction pairs, pos and neg
    train_dti_inter_mat: disease-microbe interaction mat, 1 for pos, 0 for neg used in test, -1 for other negs
    feats: input embeddings
    pos: pos pairs for contrast
    mps: meta-path network constructed previously
    nei_index: normalized sim adj matrix (DAD) for GCN
    sn_edge: similarity network edges
    mp_edge: meta-path network edges
    """

    model.eval()
    with torch.no_grad():
        """
        in test, we pretict all pairs result, and input them to the negative sampling module
        """
        row_dti_id = link_dti_id_test.permute(1, 0)[0]
        col_dti_id = link_dti_id_test.permute(1, 0)[1]
        disease_index = row_dti_id
        microbe_index = col_dti_id + test_dti_inter_mat.shape[0]
        disease_index_all = torch.arange(len(pos[0])).reshape([-1, 1])
        disease_index_all = disease_index_all.repeat(1, microbe_num).cuda()
        microbe_index_all = torch.arange(len(feats) - len(pos[0])).repeat(disease_num, 1).cuda()
        microbe_index_all = microbe_index_all + test_dti_inter_mat.shape[0]
        ADJ = test_dti_inter_mat[row_dti_id, col_dti_id]

        output, _, encoder_embeds = model(microbe_index_all, disease_index_all, feats, pos, mps, nei_index, sn_edge,
                                          mp_edge, ADJ)

        preds = encoder_embeds.reshape(disease_num, microbe_num)
        pred = preds[disease_index, microbe_index - disease_num]
        # preds = output

        targets = test_dti_inter_mat[row_dti_id, col_dti_id]
        loss_test = 0
        acc_dti_test = accuracy(pred, targets)
        if (epoch + 1) % 10 == 0:
            print("test acc: {}".format(acc_dti_test))
    return acc_dti_test, loss_test, pred, targets, encoder_embeds, encoder_embeds, row_dti_id, col_dti_id


# Train model
t_total = time.time()
acc_score = np.zeros(5)
auc_score = np.zeros(5)
aupr_score = np.zeros(5)
f1_score = np.zeros(5)
specificity_score = np.zeros(5)
fold_num = 5
fold_acc_score = np.zeros(5)
fold_auc_score = np.zeros(5)
fold_aupr_score = np.zeros(5)
fold_f1_score = np.zeros(5)
fold_mcc_score = np.zeros(5)
fold_specificity_score = np.zeros(5)
folf_pd = [[], [], [], [], []]
tprs = []
fprs = []
precisions = np.zeros(5)
recalls = np.zeros(5)
mean_fpr = np.linspace(0, 1, 548366)

"""
conduct the 5-CV operation on the dataset 

diseasefeat：inut disease features
microbefeat：input microbe features
adj：1:1 pos and neg pairs
folds：the 5-CV folds of 1:1 pos neg pairs
nodefeatures：concat disease features and microbe features
"""
disease_num, microbe_num, adj, folds, nodefeatures, positive_sample_num = cross_5_folds(args.dataset, args.seed)

int_label = adj.astype(np.int64)
groups = folds.astype(np.float32)

"""
based on the generated 5-CV adj cut the data into train and test
"""
_, _, test_train_interact_pos, test_train_label, \
    test_val_interact_pos, test_val_label, train_negative_inter_pos = first_spilt_label(int_label, groups, args.seed,
                                                                                        args.dataset, args.pos_num,
                                                                                        args.key)
test_adj_list = []

t1 = time.time()

for train_times in range(fold_num):

    test_dti_inter_mat = -np.ones((disease_num, microbe_num))  # [microbe_num, disease_num]
    for j, inter in enumerate(test_train_interact_pos[train_times]):
        microbe_id = inter[1]
        disease_id = inter[0]
        label = test_train_label[train_times][j]
        test_dti_inter_mat[disease_id][microbe_id] = label
    for j, inter in enumerate(test_val_interact_pos[train_times]):
        microbe_id = inter[1]
        disease_id = inter[0]
        label = test_val_label[train_times][j]
        test_dti_inter_mat[disease_id][microbe_id] = label
    test_dti_inter_mat = test_dti_inter_mat.tolist()

    ori_dti_inter_mat = torch.FloatTensor(test_dti_inter_mat)
    ori_train_interact_pos = torch.LongTensor(test_train_interact_pos[train_times])
    ori_val_interact_pos = torch.LongTensor(test_val_interact_pos[train_times])
    # ori_dti_inter_mat 关联矩阵
    """
    load encoder data
    """
    nei_index, feats, mps, poss, sn_edge, mp_edge = load_data(args.dataset, train_times)
    feats_dim_list = feats.shape[0]
    P = int(len(mps))

    model = MDA_Graph(pnn_hyper, Deco_hyper, microbe_num, disease_num, args.dropout, args, feats_dim_list, P=P, )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    best_test = 0

    output = torch.ones(disease_num * microbe_num)
    # --------------------------------------------------------------
    if args.cuda:
        model = model.cuda()
        ori_dti_inter_mat = ori_dti_inter_mat.cuda()
        ori_train_interact_pos = ori_train_interact_pos.cuda()
        ori_val_interact_pos = ori_val_interact_pos.cuda()

        feats = feats.cuda()
        mps = [mp.cuda() for mp in mps]
        pos = [pos.cuda() for pos in poss]
        nei_index = [nei.cuda() for nei in nei_index]
        sn_edge = [sc.cuda() for sc in sn_edge]
        mp_edge = [mp.cuda() for mp in mp_edge]

    for epoch in range(args.epochs):
        """
        construct neg samples for each iteration
        """
        dti_inter_mat = ori_dti_inter_mat
        train_interact_pos = ori_train_interact_pos
        val_interact_pos = ori_val_interact_pos

        if args.cuda:
            dti_inter_mat = dti_inter_mat.cuda()
            train_interact_pos = train_interact_pos.cuda()
            val_interact_pos = val_interact_pos.cuda()
        if (epoch + 1) % 500 == 0:
            print('Epoch: {:04d}'.format(epoch + 1), 'Fold:', train_times + 1)
        train(epoch, train_interact_pos, dti_inter_mat, feats, pos, mps, nei_index, sn_edge, mp_edge)
        if (epoch + 1) % 500 == 0:
            test_score, test_loss, predicts, targets, output, encoder_embeds, trows, tcols = test(val_interact_pos,
                                                                                                  dti_inter_mat, feats,
                                                                                                  pos, mps, nei_index,
                                                                                                  sn_edge, mp_edge)
            if test_score > best_test:
                acc_score[train_times] = round(best_test, 4)
        # print(output.shape)
    # ：final test result for each fold
    test_score, test_loss, predicts, targets, preditall, encoder_embeds, trows, tcols = test(val_interact_pos,
                                                                                             dti_inter_mat, feats, pos,
                                                                                             mps, nei_index, sn_edge,
                                                                                             mp_edge)
    fold_test = test_score
    train_times_str = str(train_times)
    # np.savetxt('GECLMDA_opt_5CV_'+args.dataset+train_times_str+'.txt',output.cpu(),delimiter=',')
    fold_acc_score[train_times] = round(accuracy(predicts, targets), 4)
    fold_auc_score[train_times] = round(auc(predicts, targets), 4)
    fold_aupr_score[train_times] = round(aupr(predicts, targets), 4)
    fold_specificity_score[train_times] = round(specificity(predicts, targets), 4)
    fold_f1_score[train_times] = round(f1(predicts, targets), 4)
    fold_mcc_score[train_times] = round(mcc(predicts, targets), 4)
    precisions[train_times] = round(precision(predicts, targets), 4)
    recalls[train_times] = round(recall(predicts, targets), 4)
    targets = targets.detach().cpu().numpy()
    trows = trows.cpu().numpy()
    tcols = tcols.cpu().numpy()
    predicts = predicts.detach().cpu().numpy()
    fpr, tpr, _ = metrics.roc_curve(targets, predicts)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    precisiono, recallo, _ = metrics.precision_recall_curve(targets, predicts)

    if train_times == 0:
        all_labels = targets
        all_preds = predicts
    else:
        all_labels = np.append(all_labels, targets)
        all_preds = np.append(all_preds, predicts)

    df = pd.DataFrame({'Column1': trows, 'Column2': tcols, 'Column3': predicts})
    folf_pd[train_times] = df
    if train_times != 4:
        print("-***************************************----")
        print("acc socre:{} avg:{}".format(fold_acc_score, np.mean(fold_acc_score)))
        print("auc socre:{} avg:{}".format(fold_auc_score, np.mean(fold_auc_score)))
        print("aupr score:{} avg:{}".format(fold_aupr_score, np.mean(fold_aupr_score)))
        print("specificity score:{} avg:{}".format(fold_specificity_score, np.mean(fold_specificity_score)))
        print("f1score:{} avg:{}".format(fold_f1_score, np.mean(fold_f1_score)))
        print("pre socre:{} avg:{}".format(precisions, np.mean(precisions)))
        print("recall socre:{} avg:{}".format(recalls, np.mean(recalls)))

t2 = time.time()
t_all = t2 - t1
print("五折交叉时间：{}".format(t_all))

with open('train_label.pkl', 'wb') as f:
    pickle.dump(folf_pd, f)

print("------------------------")

print("acc socre:{} avg:{} std{}".format(fold_acc_score, np.mean(fold_acc_score), np.std(fold_acc_score)))
print("auc socre:{} avg:{} std{}".format(fold_auc_score, np.mean(fold_auc_score), np.std(fold_auc_score)))
print("aupr score:{} avg:{} std{}".format(fold_aupr_score, np.mean(fold_aupr_score), np.std(fold_aupr_score)))
print("f1 score:{} avg:{} std{}".format(fold_f1_score, np.mean(fold_f1_score), np.std(fold_f1_score)))
print("pre socre:{} avg:{} std{}".format(precisions, np.mean(precisions), np.std(precisions)))
print("recall socre:{} avg:{} std{}".format(recalls, np.mean(recalls), np.std(recalls)))
print("specificity score:{} avg:{}".format(fold_specificity_score, np.mean(fold_specificity_score)),np.std(fold_specificity_score))

if args.savedata == 1:
    np.savetxt(f'{os.path.join(project_root,"outputs","txt")}GECLMDA_tpr_5CV_' + args.dataset + '.txt', mean_tpr, delimiter=',')
    np.savetxt(f'{os.path.join(project_root,"outputs","txt")}GECLMDA_fpr_5CV_' + args.dataset + '.txt', mean_fpr, delimiter=',')
    np.savetxt(f'{os.path.join(project_root,"outputs","txt")}GECLMDA_pre_5CV_' + args.dataset + '.txt', precisiono, delimiter=',')
    np.savetxt(f'{os.path.join(project_root,"outputs","txt")}GECLMDA_rec_5CV_' + args.dataset + '.txt', recallo, delimiter=',')
    np.savetxt(f'{os.path.join(project_root,"outputs","txt")}GECLMDA_opt_5CV_' + args.dataset + '.txt', output.cpu(), delimiter=',')
    np.savetxt(f'{os.path.join(project_root,"outputs","txt")}GECLMDA_opt_5CV_' + args.dataset + '.txt', f1_score, delimiter=',')
    np.savetxt(f'{os.path.join(project_root,"outputs","txt")}GECLMDA_ped_5CV_' + args.dataset + '.txt', predicts, delimiter=',')
    np.savetxt(f'{os.path.join(project_root,"outputs","txt")}GECLMDA_target_5CV_' + args.dataset + '.txt', targets, delimiter=',')
