import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .sn_encoder import Sn_encoder
from .mp_encoder import Mp_encoder
from .contrast import Contrast
from .sparseencoder import SparseAutoencoder

class Encoder(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P,
                  tau, lam, mp_ngcn, sn_ngcn,disease_num, microbe_num):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.feats_dim = feats_dim_list
        self.fc_list = nn.Linear(self.feats_dim, hidden_dim, bias=True)

        nn.init.xavier_normal_(self.fc_list.weight, gain=1.414)
        self.mp_ngcn = mp_ngcn
        self.sn_ngcn = sn_ngcn

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)

        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop, disease_num)
        self.sn = Sn_encoder(hidden_dim,self.sn_ngcn)
        self.se = SparseAutoencoder(hidden_dim,0.1)

        self.contrast_d = Contrast(hidden_dim, tau, lam)
        self.contrast_m = Contrast(hidden_dim, tau, lam)

    def forward(self, feats, pos, mps, nei_index, sn_edge, mp_edge):
        h_all = F.elu(self.feat_drop(self.fc_list(feats)))
        z_sn = self.sn(h_all, nei_index)
        z_mp = self.mp(h_all, mps, z_sn)

        z_mp, encoded1 = self.se(z_mp)
        z_sn, encoded = self.se(h_all)

        contrast_loss = self.contrast_d(z_mp[:len(pos[0]), :], z_sn[:len(pos[0]), :], pos[0]) + \
                        self.contrast_m(z_mp[len(pos[0]):, :], z_sn[len(pos[0]):, :], pos[1])

        sparse_loss = self.se.sparse_loss(encoded)

        total_loss = contrast_loss + sparse_loss

        return total_loss, z_mp



class NN(nn.Module):
    def __init__(self, ninput, nhidden, noutput, nlayers, dropout):

        super(NN, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.encode = torch.nn.ModuleList([
            torch.nn.Linear(ninput, noutput) for l in range(self.nlayers)])

    def forward(self, x):
        for l, linear in enumerate(self.encode):
            x = F.relu(linear(x))
        return x

class MDA_Decoder(nn.Module):
    def __init__(self, microbe_num, disease_num, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(MDA_Decoder, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.disease_num = disease_num
        self.microbe_num = microbe_num
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else Nodefeat_size, Nodefeat_size) for l in
            range(3)])

        # self.linear0 = torch.nn.Linear(Nodefeat_size, Nodefeat_size)
        self.linear = torch.nn.Linear(Nodefeat_size, 1)
        self.disease_linear = torch.nn.Linear(Nodefeat_size, Nodefeat_size)
        self.microbe_linear = torch.nn.Linear(Nodefeat_size, Nodefeat_size)


    def forward(self, nodes_features, disease_index, microbe_index,ADJ):

        microbe_features = nodes_features[microbe_index]
        disease_features = nodes_features[disease_index]
        # print('microbe_features', microbe_index.shape)
        # print('disease_features',disease_index.shape)
        microbe_features = self.microbe_linear(microbe_features)
        disease_features = self.disease_linear(disease_features)
        # print('microbe_features',microbe_features.shape)
        # print('disease_features',disease_features.shape)
        # microbe_features = self.node_level[0](microbe_features, ADJ)
        # disease_features = self.node_level[1](disease_features, ADJ)
        pair_nodes_features = disease_features*microbe_features
        # print('disease_features',pair_nodes_features.shape)
        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))

        pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
        output = self.linear(pair_nodes_features)
        # output = self.linear(output)
        return output,torch.sigmoid(output)

class MDA_Graph(nn.Module):

    def __init__(self,  PNN_hyper, DECO_hyper, microbe_num, disease_num, dropout, args_pre, feats_dim_list, P, train_type=1):
        super(MDA_Graph, self).__init__()
        # self.mp_nn = NN(PNN_hyper[0], PNN_hyper[1], PNN_hyper[2], PNN_hyper[3], dropout)
        # self.sn_nn = NN(PNN_hyper[0], PNN_hyper[1], PNN_hyper[2], PNN_hyper[3], dropout)
        self.MDA_Decoder = MDA_Decoder(microbe_num, disease_num, DECO_hyper[0], DECO_hyper[0], DECO_hyper[2], dropout)
        self.LayerNorm = torch.nn.LayerNorm(DECO_hyper[0])

        self.encoder = Encoder(args_pre.hidden_dim, feats_dim_list, args_pre.feat_drop, args_pre.attn_drop,
                 P,  args_pre.tau, args_pre.lam, args_pre.mp_ngcn,args_pre.sn_ngcn,disease_num, microbe_num)
        self.encoder.cuda()
        self.disease_num = disease_num
        self.proein_num = microbe_num






    def forward(self, microbe_index, disease_index, feats, pos, mps, nei_index, sn_edge, mp_edge,CAD):
        # 1. 编码器阶段（包含对比学习）
        loss,embs_sn= self.encoder(feats, pos, mps, nei_index, sn_edge, mp_edge)


        embs_sn = self.LayerNorm(embs_sn)

        output,simoutput = self.MDA_Decoder(embs_sn, disease_index, microbe_index,CAD)
        output = output.view(-1)
        simoutput= simoutput.view(-1)


        return output, loss, simoutput


