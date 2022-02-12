from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from model.base_model import GNNModel
from model.Layers import MultiLayerPerceptron, SimpleHypergraphConv
from utility import logger


class HGMBR(GNNModel):
    def __init__(self, model_cfg, data_manager) -> None:
        super().__init__(model_cfg, data_manager)
        self.num_action = data_manager.data_info["actions_num"]
        self.relation = list(self.data.train_df.columns)[2:]
        self.action_embeds = nn.Parameter(torch.FloatTensor(self.num_action, self.emb_dim), requires_grad=True)

        self.beta = model_cfg.beta

        nn.init.xavier_normal_(self.action_embeds)
        laplacian_matrix = self.get_lap_mat()
        self.register_buffer('laplacian_matrix', laplacian_matrix, persistent=False)

        self.num_node = self.num_user + self.num_item + self.num_action
        self.hypergraph_channels = self._construct_hypergraphs()
        logger.info(f"hypergraphs constructed! {self.hypergraph_channels}")
        self.dropout = nn.Dropout(self.mess_dropout, True)
        self.train_mat_dropout = nn.Dropout(self.edge_dropout, True)
        self.activation = nn.LeakyReLU()

        self.graph_conv_layers = nn.ModuleList([])

        for channel in range(len(self.hypergraph_channels)):
            self.graph_conv_layers.append(nn.ModuleList([]))
            for layer in range(self.num_layer):
                self.graph_conv_layers[channel].append(
                    SimpleHypergraphConv(edge_dropout=self.edge_dropout)
                )

        self.action_classifier = MultiLayerPerceptron(self.emb_dim, [], self.mess_dropout)
        conversion_statistics = torch.FloatTensor(self.get_conversion(self.data.train_df)).unsqueeze(-1)
        self.register_buffer("action_label", conversion_statistics, persistent=False)
        self.ssl_loss = nn.MSELoss()

        self.cwd = nn.Parameter(torch.FloatTensor([1] * len(self.hypergraph_channels)))

    def get_lap_mat(self):
        train_data_desired = self.data.train_df.loc[
            self.data.train_df[self.data.cfg.data.desired_action] == 1, ['user_id', 'photo_id']].values
        return self._get_lap_mat(train_data_desired)



    def propagate(self, *args, **kwargs):
        all_emb = torch.cat((self.u_embeds, self.i_embeds, self.action_embeds), dim=0)
        channel_emb_all = []
        for channel in range(len(self.hypergraph_channels)):
            channel_emb = all_emb
            graph_name = self.hypergraph_channels[channel]
            graph_channel = self.get_buffer(graph_name)
            for layer in range(self.num_layer):
                conv_layer = self.graph_conv_layers[channel][layer]
                channel_emb = self.dropout(conv_layer(graph_channel, channel_emb))
            channel_emb_all.append(channel_emb)

        # channel_embed_fusion = torch.cat(channel_emb_all, dim=1)
        channel_embed_fusion = torch.stack(channel_emb_all, dim=-1)
        cwd = self.cwd / torch.sum(self.cwd)
        channel_embed_fusion = torch.sum(channel_embed_fusion * cwd, dim=-1)

        user_embed = channel_embed_fusion[:self.num_user]
        photo_embed = channel_embed_fusion[self.num_user:self.num_user + self.num_item]
        action_embed = channel_embed_fusion[self.num_user + self.num_item:]

        ui_embed = torch.cat((self.u_embeds, self.i_embeds), dim=0)
        ego_embeddings = torch.mm(self.laplacian_matrix, ui_embed)

        u_neighbour_embeds, i_neighbour_embeds = torch.split(ego_embeddings, (self.num_user, self.num_item), 0)
        user_embed = torch.cat((user_embed, u_neighbour_embeds), dim=1)
        photo_embed = torch.cat((photo_embed, i_neighbour_embeds), dim=1)

        return user_embed, photo_embed, action_embed

    def get_conversion(self, df):
        statistics = [1]
        desired_action_num = df[df[self.data.cfg.data.desired_action] == 1].shape[0]
        logger.info(f"desired action num: {desired_action_num}")
        for action in self.relation[1:]:
            df_action = df[df[action] == 1]
            total_action = df_action.shape[0]
            desired_action = df_action[df_action[self.data.cfg.data.desired_action] == 1].shape[0]
            statistics.append(desired_action / total_action)
            logger.info(f"{action}: {desired_action}/{total_action}")
        return statistics

    def forward(self, users, POIs):
        user_embed, photo_embed, action_embed = self.propagate()
        POIs_embedding = photo_embed[POIs]
        users_embedding = user_embed[users].expand(-1, POIs.shape[1], -1)

        pred = self.predict(users_embedding, POIs_embedding)
        pred = F.softmax(pred, dim=-1)
        loss = self.regularize(users_embedding, POIs_embedding, action_embed)
        return pred, loss

    def regularize(self, users_feature, POIs_feature, action_embedding):
        '''
        embeddings of targets for predicting -> extra loss(default: L2 loss...)
        '''
        batch_size = users_feature.shape[0]
        l2_loss = ((users_feature ** 2).sum() + (POIs_feature ** 2).sum()) / batch_size

        action_score = self.action_classifier(action_embedding)
        ssl_loss = self.ssl_loss(action_score, self.action_label)
        # print(action_score, self.action_label, ssl_loss)

        loss = self.lam * l2_loss + self.beta * ssl_loss
        loss_dict = {"reg_loss": loss, "l2_loss": l2_loss, "l2_loss_scaled": self.lam * l2_loss,
                     "ssl_loss": ssl_loss, "ssl_loss_scaled": self.beta * ssl_loss}
        return loss_dict

    def evaluate(self, users, POIs=None, *args, **kwargs):
        if POIs is not None:
            user_embed, photo_embed, action_embed = self.propagate()
            POIs_embedding = photo_embed[POIs]
            users_embedding = user_embed[users].expand(-1, POIs.shape[1], -1)

            pred = self.predict(users_embedding, POIs_embedding)
            loss = self.regularize(users_embedding, POIs_embedding, action_embed)
            return pred, loss
        else:
            users_feature, POIs_feature, action_embed = self.propagate()
            user_feature = users_feature[users]
            scores = torch.mm(user_feature, POIs_feature.t())
            return scores, 0

    def _construct_hypergraphs(self):

        valid_idx = self.data.train_df[self.data.cfg.data.desired_action] == 1
        invalid_idx = self.data.train_df[self.data.cfg.data.desired_action] == 0

        train_table_bi = self.data.train_df.loc[valid_idx, ['user_id', 'photo_id']].values
        graph_name = "hypergraph_cf"
        self.register_buffer(graph_name, self._construct_single_hypergraph(train_table_bi), persistent=False)
        hypergraph_channels = [graph_name]

        train_table_valid = self.data.train_df[valid_idx].values
        graph_name = "hypergraph_c"
        self.register_buffer(graph_name, self._construct_single_hypergraph(train_table_valid), persistent=False)
        hypergraph_channels.append(graph_name)

        train_table_invalid = self.data.train_df[invalid_idx].values
        graph_name = "hypergraph_a"
        self.register_buffer(graph_name, self._construct_single_hypergraph(train_table_invalid), persistent=False)
        hypergraph_channels.append(graph_name)
        return hypergraph_channels

    def _construct_single_hypergraph(self, table, conversion_rate=None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        hyperedge_idx, col_idx = table.nonzero()
        hyperedge_idx_all = []
        node_idx_all = []

        # User entries
        user_entries_idx = (col_idx == 0)
        hyperedge_idx_user = hyperedge_idx[user_entries_idx]
        user_idx_entry = table[hyperedge_idx_user, col_idx[user_entries_idx]]
        hyperedge_idx_all.append(hyperedge_idx_user)
        node_idx_all.append(user_idx_entry)

        # Photo entries
        photo_entries_idx = (col_idx == 1)
        hyperedge_idx_photo = hyperedge_idx[photo_entries_idx]
        photo_idx_entry = table[hyperedge_idx_photo, col_idx[photo_entries_idx]] + self.num_user
        hyperedge_idx_all.append(hyperedge_idx_photo)
        node_idx_all.append(photo_idx_entry)

        # Action entries
        for action in range(self.num_action):
            action_entries_idx = (col_idx == action + 2)
            hyperedge_idx_action = hyperedge_idx[action_entries_idx]
            action_idx_entry = np.full_like(hyperedge_idx_action, action) + self.num_user + self.num_item
            hyperedge_idx_all.append(hyperedge_idx_action)
            node_idx_all.append(action_idx_entry)

        hyperedge_idx_all = np.concatenate(hyperedge_idx_all)
        node_idx_all = np.concatenate(node_idx_all)

        indices = torch.from_numpy(np.stack([node_idx_all, hyperedge_idx_all]).astype(np.int64))
        values = torch.ones(indices.shape[1])
        matrix_shape = (self.num_node, len(table))

        graph = torch.sparse_coo_tensor(indices, values, size=matrix_shape)
        graph_t = graph.t()
        node_degree = torch.sparse.sum(graph, dim=1).to_dense()
        edge_degree = torch.sparse.sum(graph_t, dim=1).to_dense()

        D_e = sparse_diag(torch.nan_to_num(1 / edge_degree, nan=0, posinf=0, neginf=0))
        D_v_sqrt = sparse_diag(torch.sqrt(torch.nan_to_num(1 / node_degree, nan=0, posinf=0, neginf=0)))

        if conversion_rate is not None:
            conversion_rate = torch.FloatTensor(conversion_rate).unsqueeze(0)
            weight_array = torch.sum(torch.from_numpy(table[:, 2:]).float() * conversion_rate, dim=1)
            W = sparse_diag(weight_array)
        else:
            W = sparse_diag(torch.ones(matrix_shape[1]))

        fast_graph = torch.sparse.mm(D_v_sqrt, graph)
        fast_graph = torch.sparse.mm(fast_graph, W)
        fast_graph = torch.sparse.mm(fast_graph, D_e)
        fast_graph = torch.sparse.mm(fast_graph, graph_t)
        fast_graph = torch.sparse.mm(fast_graph, D_v_sqrt)

        return fast_graph

    def eliminate_small(self, x, threshold=None):
        if threshold is None:
            threshold = 1 / self.num_node
        print(
            f'max value: {torch.max(x._values())}, min value: {torch.min(x._values())}, mean value: {torch.mean(x._values())}')
        print(f'benchmark: {threshold}')
        mask = (x._values() > threshold).nonzero().view(-1)
        nv = x._values().index_select(0, mask)
        ni = x._indices().index_select(1, mask)
        return torch.sparse_coo_tensor(ni, nv, size=x.shape)

    @property
    def dataset_format(self):
        return "SamplingTrainSet"


def sparse_diag(values):
    indices = torch.arange(0, values.shape[0], dtype=torch.long, device=values.device).expand(2, -1)
    return torch.sparse.FloatTensor(indices, values, (values.shape[0], values.shape[0]))
