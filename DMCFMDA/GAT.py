import torch.nn as nn
import torch as th
import torch_geometric.nn as pt


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim, num_heads, dropout):
        """
        GAT module for Graph Transformer and GAT layers
        :param input_dim: Input feature dimension
        :param hidden_dims: List of hidden dimensions for GAT layers
        :param out_dim: Output feature dimension
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate
        """
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.elu = nn.ELU()
        in_dim = input_dim

        for idx, hidden_dim in enumerate(hidden_dims):
            out_heads = num_heads if idx < len(hidden_dims) - 1 else 1
            layer_out_dim = out_dim if idx == len(hidden_dims) - 1 else hidden_dim
            # This module was originally written against DGL's GAT API.  The
            # fixed-fold adapter uses the PyG implementation, whose argument
            # names are ``heads`` and ``dropout``.
            self.layers.append(pt.GATConv(in_dim, layer_out_dim, heads=out_heads, dropout=dropout))
            in_dim = layer_out_dim * out_heads

        self.residual_layer = nn.Linear(input_dim, out_dim)
        self.fuse_weight = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(0.5)

    def forward(self, graph, features, residual_features):
        """
        Forward pass for GAT
        :param graph: Input graph
        :param features: Node features
        :param residual_features: Residual connection features
        :return: Output embeddings
        """
        x = features
        for layer in self.layers:
            # PyG GATConv takes (x, edge_index), unlike DGL layers which use
            # (graph, x).
            x = layer(x, graph)
            x = x.view(x.size(0), -1)
            x = self.elu(x)

        x = self.fuse_weight * x + (1 - self.fuse_weight) * self.elu(self.residual_layer(residual_features))
        return x
