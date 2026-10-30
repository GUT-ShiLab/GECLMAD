import dgl
import math
import random
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return th.LongTensor(edge_index)


def make_adj(edges, size):
    edges_tensor = th.LongTensor(edges).t()
    values = th.ones(len(edges))
    adj = th.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    # adj_dense=adj
    # adj_dense.long()
    return adj


def predict_case(data, args):
    data['m_d_matrix'] = make_adj(data['m_d'], (args.microbe, args.disease_number))
    m_d_matrix = data['m_d_matrix']
    one_index = []
    zero_index = []
    for i in range(m_d_matrix.shape[0]):
        for j in range(m_d_matrix.shape[1]):
            if m_d_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)

    random.shuffle(one_index)
    one_index = np.array(one_index)
    random.shuffle(zero_index)
    zero_index = np.array(zero_index)

    train = np.concatenate(
        (one_index, zero_index[:int(args.negative_rate * len(one_index))]))
    mm = data['mm_f'] * np.where(data['mm_f'] == 0, 0, 1) + get_gaussian(data['m_d_matrix']) * np.where(
        data['mm_f'] == 1, 0, 1)
    dd = data['dd_s'] * np.where(data['dd_s'] == 0, 0, 1) + get_gaussian(data['m_d_matrix'].t()) * np.where(
        data['dd_s'] == 1, 0, 1)
    data['mm'] = {'data_matrix': mm, 'edges': get_edge_index(mm)}
    data['dd'] = {'data_matrix': dd, 'edges': get_edge_index(dd)}
    data['train'] = train


def data_processing(data, args):
    """Prepare DMCFMDA's native fixed priors and global-CV candidates.

    The graph remains fixed across folds, but DMCFMDA's own similarity rule is
    retained: preserve a nonzero supplied similarity and fill its zero entries
    with Gaussian interaction-profile similarity from that fixed graph.
    """
    labels = np.zeros((args.microbe, args.disease_number), dtype=np.int64)
    labels[data['md'][:, 0], data['md'][:, 1]] = 1
    data['labels'] = labels
    data['topology_md'] = data['md'].copy()
    microbe_gip = get_gaussian(labels)
    disease_gip = get_gaussian(labels.T)
    data['ms'] = (
        data['mf'] * np.where(data['mf'] == 0, 0, 1)
        + microbe_gip * np.where(data['mf'] == 1, 0, 1)
    )
    data['ds'] = (
        data['dss'] * np.where(data['dss'] == 0, 0, 1)
        + disease_gip * np.where(data['dss'] == 1, 0, 1)
    )
    assignment = np.loadtxt(args.data_dir + 'fold_assignments.txt', dtype=int)
    if assignment.ndim == 1:
        assignment = assignment.reshape(1, -1)
    data['fold_assignment'] = np.full(labels.shape, -1, dtype=np.int64)
    data['fold_assignment'][assignment[:, 0], assignment[:, 1]] = assignment[:, 2]
    if not np.array_equal((data['fold_assignment'] >= 0), labels.astype(bool)):
        raise ValueError('fold_assignments.txt does not match knowinteraction.txt')

def k_matrix(matrix, k=25):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)

def get_data(args):
    data = dict()

    mf = np.loadtxt(args.data_dir + 'km.txt', dtype=float, delimiter=',')    #M-D

    dss = np.loadtxt(args.data_dir + 'ID.txt', dtype=float)    #M-D

    data['microbe'] = int(mf.shape[0])
    data['disease_number'] = int(dss.shape[0])
    data['mf'] = mf
    data['dss'] = dss
    data['md'] = np.loadtxt(args.data_dir + 'knowinteraction.txt', dtype=int) - 1   # M-D
    if data['md'].ndim == 1:
        data['md'] = data['md'].reshape(1, -1)

    return data


def get_gaussian(adj):
    Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float32)
    gamaa = 1
    sumnorm = 0
    for i in range(adj.shape[0]):
        norm = np.linalg.norm(adj[i]) ** 2
        sumnorm = sumnorm + norm
    gama = gamaa / (sumnorm / adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))

    return Gaussian
