import os
from abc import ABC

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from utility import logger
from hydra.utils import to_absolute_path


class Model(nn.Module, ABC):
    """
    base class for all MF-based model
    packing embedding initialization, embedding choosing in forward

    NEED IMPLEMENT:
    - `propagate`: all raw embeddings -> processed embeddings(user/POI)
    - `predict`: processed embeddings of targets(users/POIs inputs) -> scores

    OPTIONAL:
    - `regularize`: processed embeddings of targets(users/POIs inputs) -> extra loss(default: L2)
    """

    def __init__(self, model_cfg: DictConfig, data_manager) -> None:
        super().__init__()
        self.data = data_manager
        self.data_info = data_manager.data_info
        self.model_cfg = model_cfg
        self.num_user = data_manager.data_info["user_ids_num"]
        self.num_item = data_manager.data_info["photo_ids_num"]
        self.lam = model_cfg.lam
        self.emb_dim = model_cfg.emb_dim

        if not self.model_cfg.embed_pretrain:
            logger.info("Create embedding...")
            self.u_embeds = nn.Parameter(
                torch.FloatTensor(self.num_user, self.emb_dim), requires_grad=True)
            self.i_embeds = nn.Parameter(
                torch.FloatTensor(self.num_item, self.emb_dim), requires_grad=True)
            nn.init.xavier_normal_(self.u_embeds)
            nn.init.xavier_normal_(self.i_embeds)

        else:
            embed_path = self.model_cfg.embed_path
            if self.model_cfg.embed_path.startswith("~"):
                embed_path = os.path.expanduser(embed_path)
            embed_path = to_absolute_path(embed_path)
            logger.info(f"Load embedding from {embed_path}...")
            load_data = torch.load(os.path.join(embed_path, 'model.pt'), map_location='cpu')
            self.u_embeds = nn.Parameter(F.normalize(load_data['u_embeds']))
            self.i_embeds = nn.Parameter(F.normalize(load_data['i_embeds']))

    def propagate(self, *args, **kwargs):
        """
        raw embeddings -> embeddings for predicting
        return (user's,POI's)
        """
        raise NotImplementedError

    def predict(self, users_feature, POIs_feature, *args, **kwargs):
        return torch.sum(users_feature * POIs_feature, dim=-1)

    def regularize(self, users_feature, POIs_feature, *args, **kwargs):
        """
        embeddings of targets for predicting -> extra loss(default: L2 loss...)
        """
        batch_size = users_feature.shape[0]
        l2_loss = ((users_feature ** 2).sum() + (POIs_feature ** 2).sum()) / batch_size
        loss = self.lam * l2_loss
        loss_dict = {"reg_loss": loss, "l2_loss": l2_loss, "l2_loss_scaled": self.lam * l2_loss}
        return loss_dict

    def forward(self, users, POIs, *args, **kwargs):
        users_feature, POIs_feature = self.propagate(*args, **kwargs)
        POIs_embedding = POIs_feature[POIs]
        users_embedding = users_feature[users].expand(- 1, POIs.shape[1], -1)
        pred = self.predict(users_embedding, POIs_embedding)
        loss = self.regularize(users_embedding, POIs_embedding)
        return pred, loss

    def evaluate(self, users, POIs=None, *args, **kwargs):
        if POIs is not None:
            return self.forward(users, POIs, *args, **kwargs)
        else:
            users_feature, POIs_feature = self.propagate()
            user_feature = users_feature[users]
            scores = torch.mm(user_feature, POIs_feature.t())
            return scores, 0

    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, 'model.pt'))

    def load(self, path):
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if path.endswith(".pt") or path.endswith(".pth"):
            self.load_state_dict(torch.load(path), strict=False)
        else:
            self.load_state_dict(torch.load(os.path.join(path, 'model.pt')), strict=False)
        logger.info(f"Load model from {path}")

    @property
    def dataset_format(self):
        """
        return the format of training data
        """
        raise NotImplementedError


class GNNModel(Model, ABC):
    def __init__(self, model_cfg, data_manager) -> None:
        super().__init__(model_cfg, data_manager)
        self.num_layer = model_cfg.num_layer
        self.mess_dropout = model_cfg.mess_dropout
        self.edge_dropout = model_cfg.edge_dropout

    def _get_idx_mat(self, train_data, normalize=False):
        value_fill_mat = np.ones(train_data.shape[0])
        shape = (self.num_user, self.num_item)
        train_mat = sp.coo_matrix((value_fill_mat, (train_data[:, 0], train_data[:, 1])),
                                  shape=shape, dtype=np.float32)
        if normalize:
            train_mat = self.normalize(train_mat)
        train_mat = train_mat.todok()
        return self.sparse_mx_to_torch_sparse_tensor(train_mat)

    @staticmethod
    def normalize(mx: sp.spmatrix) -> sp.spmatrix:
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        np.seterr(divide='ignore')
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    @staticmethod
    def normalize_sym(adj: sp.spmatrix) -> sp.spmatrix:
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        np.seterr(divide='ignore')
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    @staticmethod
    def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.sparse_coo_tensor:
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    @staticmethod
    def get_fast_hypergraph(index, size, normalize=False):
        """
        Get the hypergraph of the given index.
        param index: the index of the hypergraph, np.array
        param size: size of incidence matrix, (num_node, num_edge)
        param normalize: whether to normalize the hypergraph
        return: H * H.T
        """
        hypergraph_idx = torch.LongTensor(index)
        values = torch.ones(index.shape[1], dtype=torch.float)
        graph = torch.sparse_coo_tensor(hypergraph_idx, values, size)
        graph_t = graph.t()
        fast_graph = torch.sparse.mm(graph, graph_t)
        if normalize:
            node_degree = torch.sparse.sum(fast_graph, dim=-1).to_dense().unsqueeze(-1)
            fast_graph = fast_graph.to_dense() / (node_degree + 1e-8)
        return fast_graph

    def _get_lap_mat(self, train_data_desired):
        # train_data_desired = self.data.train_df.loc[
        #     self.data.train_df[self.data.cfg.data.desired_action] == 1, ['user_id', 'photo_id']].values

        idx_upper_half = train_data_desired
        idx_upper_half[:, 1] = idx_upper_half[:, 1] + self.num_user

        idx_lower_half = idx_upper_half[:, [1, 0]]

        train_data = np.concatenate((idx_upper_half, idx_lower_half), axis=0)
        value_fill_mat = np.ones(train_data.shape[0])
        shape = (self.num_user + self.num_item, self.num_user + self.num_item)

        train_mat = sp.coo_matrix((value_fill_mat, (train_data[:, 0], train_data[:, 1])),
                                  shape=shape, dtype=np.float32)
        train_mat = self.normalize(train_mat)
        train_mat = train_mat.todok()
        return self.sparse_mx_to_torch_sparse_tensor(train_mat)
