import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        # attn_curr = self.attn_drop(self.att)
        attn_curr = self.att
        for embed in embeds:
            sp = self.tanh(F.dropout(self.fc(embed), 0.95)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)

        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]

        return z_mp


class Sn_encoder(nn.Module):
    def __init__(self, hidden_dim, sc_ngcn):
        super(Sn_encoder, self).__init__()
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(sc_ngcn)])
        self.sc_ngcn = sc_ngcn


    def forward(self, nei_h, nei_index):
        embeds = []

        embeds.append(nei_h)
        for i in range(self.sc_ngcn):
            embeds.append(self.node_level[i](embeds[-1], nei_index[0]))
        # print(embeds[-1].shape)
        return embeds[-1]
