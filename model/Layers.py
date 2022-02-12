import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            # layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        # print(layers)
        layers.append(nn.Sigmoid())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        # print(x.shape)
        return self.mlp(x)


class SimpleHypergraphConv(nn.Module):
    """
    Hypergraph convolution layer.
    """

    def __init__(self, edge_dropout=0.0):
        super().__init__()
        self.edge_dropout = nn.Dropout(edge_dropout)

    def forward(self, graph, x):
        indices = graph._indices()
        values = graph._values()
        values = self.edge_dropout(values)
        graph_drop = torch.sparse_coo_tensor(indices, values, size=graph.shape)
        x = torch.mm(graph_drop, x)
        return x
