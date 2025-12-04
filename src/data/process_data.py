import os
import argparse
import random
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from src.utils.load_data import preprocess_adj
from pathlib import Path

from src.data.data_reformat import *

project_root = Path(__file__).parent.parent.parent



def cross_5_folds(dataset, seed):
    random.seed(seed)
    np.random.seed(seed)

    #### 文件读取
    path_raw =  project_root / "data" / "raw"
    parent_dir = path_raw / dataset
    path_diseasefeat = os.path.join(parent_dir, "diseasefeatures.txt")
    path_microbefeat = os.path.join(parent_dir, "microbefeatures.txt")
    diseasefeat = np.loadtxt(path_diseasefeat)
    microbefeat = np.loadtxt(path_microbefeat)
    path_adj = os.path.join(parent_dir, "adj.txt")
    adj = np.loadtxt(path_adj)

    for i in adj:
        i[0] -= 1
        i[1] -= 1
    adj = adj.astype("int")
    D = len(diseasefeat)
    M = len(microbefeat)
    dm = np.array(adj)
    dm = sp.csr_matrix((dm[:, 2], (dm[:, 0], dm[:, 1])), shape=(D, M)).toarray()
    folds = np.ones(adj.shape[0])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf = kf.split(folds)
    a = []

    for i, data in enumerate(kf):
        a.append(data[1])

    for i in range(5):
        for j in range(len(a[i])):
            folds[a[i][j]] = i

    adj = adj.astype(np.int32).tolist()
    pos_num = len(adj)

    """
    randomly generate neg samples for final test
    """

    temp_adj = []
    for i in range(D):
        for j in range(M):
            value = dm[i, j]
            temp_adj.append((i, j, value))
    fold_size = len(temp_adj) // 5
    fold_size
    indices = list(range(len(temp_adj)))
    random.shuffle(indices)
    folds1 = np.empty(len(temp_adj), dtype=int)
    for idx, val in enumerate(indices):
        folds1[val] = idx // fold_size
    folds1[folds1==5]=0

    folds1 = np.array(folds1)
    temp_adj = np.array(temp_adj)
    nodefeatures = np.vstack((np.hstack((diseasefeat, dm)), np.hstack((dm.transpose(), microbefeat))))
    path_processed = project_root / "data" / "processed"/dataset
    path_processed.mkdir(parents=True, exist_ok=True)
    path_data = os.path.join(path_processed,"data_"+dataset+".npz")
    np.savez(path_data, diseasefeat=diseasefeat, microbefeat=microbefeat, adj=temp_adj, folds=folds1, nodefeat=nodefeatures)
    diseasenum = diseasefeat.astype(np.float32)
    microbenum = microbefeat.astype(np.float32)
    microbenum = len(microbenum)
    diseasenum = len(diseasenum)
    return diseasenum, microbenum, temp_adj, folds1, nodefeatures, pos_num



def first_spilt_label(inter, groups, seed, dataset,pos_num,key):
    """
    inter: 1:1 pos and neg pairs
    groups: the 5-CV folds of 1:1 pos neg pairs
    """
    random.seed(seed)
    np.random.seed(seed)

    """
    inter_folds: According to the groups division, divide inter into five group
    label_folds = the label of pairs in inter_folds; 1 for pod, 0 for neg
    pos_inter_folds = the pos pairs in each group
    neg_inter_folds = the neg pairs in each group
    """

    inter_folds = [[],[],[],[],[]]
    label_folds = [[],[],[],[],[]]
    pos_inter_folds = [[],[],[],[],[]]
    neg_inter_folds = [[],[],[],[],[]]

    for i, inter_k in enumerate(inter):
        inter_type = inter_k[-1]  # 1 positive, 0 negative
        microbe_node_id = inter_k[1]
        disease_node_id = inter_k[0]
        fold_id = int(groups[i])
        inter_folds[fold_id].append([disease_node_id, microbe_node_id])
        label_folds[fold_id].append(inter_type)
        if inter_type == 1:
            #  positive sample
            pos_inter_folds[fold_id].append([disease_node_id, microbe_node_id])
        elif inter_type == 0:
            # negative sample
            neg_inter_folds[fold_id].append([disease_node_id, microbe_node_id])
        else:
            print("inter type has problem")

    train_positive_inter_pos = [[],[],[],[],[]]
    val_positive_inter_pos = [[],[],[],[],[]]
    train_negative_inter_pos = [[],[],[],[],[]]
    val_negative_inter_pos = [[],[],[],[],[]]

    train_interact_pos = [[],[],[],[],[]]
    val_interact_pos = [[],[],[],[],[]]
    train_label = [[],[],[],[],[]]
    val_label = [[],[],[],[],[]]

    for i in range(5):
        val_fold_id = i
        for j in range(5):
            if j != val_fold_id:
                train_positive_inter_pos[i] += pos_inter_folds[j]
                train_negative_inter_pos[i] += neg_inter_folds[j]

                train_interact_pos[i] += inter_folds[j]
                train_label[i] += label_folds[j]
            else:
                val_positive_inter_pos[i] += pos_inter_folds[j]
                val_negative_inter_pos[i] += neg_inter_folds[j]

                val_interact_pos[i] += inter_folds[j]
                val_label[i] += label_folds[j]
        """
            shuffle the data
        """
        train_interact_pos[i] = np.array(train_interact_pos[i])
        train_label[i] = np.array(train_label[i])
        shuffle_train = np.arange(0,len(train_interact_pos[i]))
        random.shuffle(shuffle_train)
        train_interact_pos[i] = train_interact_pos[i][shuffle_train]
        train_label[i] = train_label[i][shuffle_train]

        val_interact_pos[i] = np.array(val_interact_pos[i])
        val_label[i] = np.array(val_label[i])
        shuffle_test = np.arange(0,len(val_interact_pos[i]))
        random.shuffle(shuffle_test)
        val_interact_pos[i] = val_interact_pos[i][shuffle_test]
        val_label[i] = val_label[i][shuffle_test]




    """
    get the data needed for encoder
    only need to run once for each dataset
    """
    if key==1:
        processdata_encoder(dataset, train_positive_inter_pos, pos_num)

    """
    train_positive_inter_pos: pos pairs in each train set for 5-fold cv
    val_positive_inter_pos: pos pairs in each test set for 5-fold cv
    train_interact_pos: pos and neg pairs in each train set for 5-fold cv
    train_label: the labels for pairs in train_interact_pos
    val_interact_pos: pos and neg pairs in each test set for 5-fold cv
    val_label: the labels for pairs in val_interact_pos
    train_negative_inter_pos: neg pairs in each train set for 5-fold cv
    """

    return train_positive_inter_pos, val_positive_inter_pos, train_interact_pos, train_label, val_interact_pos, val_label, train_negative_inter_pos




def constr_adj(node_num, interact_index, neg_inter_index, disease_num):
    """
    DAD
    """
    interact_index = np.array(interact_index)
    adj = sp.csr_matrix((np.ones((len(interact_index))), (interact_index[:, 0], interact_index[:, 1]+disease_num)), shape=(node_num, node_num)).toarray()
    adj = adj + adj.transpose()
    adj = preprocess_adj(adj)
    return adj


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='diseasevirus',  # diseasevirus  enzyme
#                         help='dataset')
#     parser.add_argument('--start_epoch', type=int, default=0,
#                         help='epoch_num')
#     parser.add_argument('--end_epoch', type=int, default=60,
#                         help='epoch_num')
#     parser.add_argument('--data_root', type=str, default='./data',
#                         help='data root')
#     parser.add_argument('--crossval', type=int, default=1,
#                         help='whether generate 5 fold')
#     parser.add_argument('--seed', type=int, default=1,
#                         help='random seed')
#     args = parser.parse_args()
#     seed = args.seed
#     random.seed(seed)
#     # load_data(seed, args.data_root, dataset=args.dataset, start_epoch=args.start_epoch, end_epoch=args.end_epoch, crossval=args.crossval)
