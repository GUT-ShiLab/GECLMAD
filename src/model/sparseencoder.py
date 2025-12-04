import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, sparsity_ratio):
        super(SparseAutoencoder, self).__init__()

        # self.encoder = nn.Linear(input_size,64)
        # self.decoder = nn.Linear(64, input_size)
        self.sparsity_ratio = sparsity_ratio
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, input_size),

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def sparse_loss(self, encoded):
        sparsity = torch.mean(encoded, dim=0)
        kl_div = self.sparsity_ratio * torch.log(self.sparsity_ratio / sparsity) + \
                 (1 - self.sparsity_ratio) * torch.log((1 - self.sparsity_ratio) / (1 - sparsity))
        # print(sparsity)
        # print(torch.log(self.sparsity_ratio / sparsity))
        return kl_div.sum()


class VGAE(nn.Module):
    def __init__(self, input_size):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(input_size, input_size)
        self.gcn_mean = GraphConvSparse(input_size, input_size, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(input_size, input_size, activation=lambda x: x)
        self.input_size = input_size

    def encode(self, X, adj):
        hidden = self.base_gcn(X, adj)
        sampled_z = self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(X.size(0), self.input_size).cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, adj):
        Z = self.encode(X, adj)
        # A_pred = dot_product_decode(Z)
        return Z


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

# class GAE(nn.Module):
# 	def __init__(self,adj):
# 		super(GAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def encode(self, X):
# 		hidden = self.base_gcn(X)
# 		z = self.mean = self.gcn_mean(hidden)
# 		return z

# 	def forward(self, X):
# 		Z = self.encode(X)
# 		A_pred = dot_product_decode(Z)
# 		return A_pred
