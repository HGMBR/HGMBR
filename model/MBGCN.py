import os
from itertools import combinations, chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import GNNModel
from time import time
from utility import logger


def sparse_diag(values):
    indices = torch.arange(0, values.shape[0], dtype=torch.long, device=values.device).expand(2, -1)
    return torch.sparse.FloatTensor(indices, values, (values.shape[0], values.shape[0]))


class MBGCN(GNNModel):
    def __init__(self, model_cfg, data_manager) -> None:
        super().__init__(model_cfg, data_manager)
        self.data_info = self.data.data_info
        self.relation = list(self.data.train_df.columns)[2:]

        try:
            relation_dict, item_graph = self.load_graph_data()
            logger.info('Load graph data successfully')
        except FileNotFoundError:
            logger.info('No graph data found, start to build graph data')
            start_time = time()
            relation_dict, item_graph = self.__create_relation_matrix(self.data.train_df)
            logger.info('Build graph data successfully, time cost: %.2f' % (time() - start_time))
            self.save_graph_data(relation_dict, item_graph)
        self.relation_dict, self.item_graph = dict(), dict()
        for key in relation_dict:
            self.register_buffer("relation_dict_" + key, relation_dict[key], persistent=False)
            self.relation_dict[key] = "relation_dict_" + key
        for key in item_graph:
            self.register_buffer("item_graph_" + key, item_graph[key], persistent=False)
            self.item_graph[key] = "item_graph_" + key

        train_data = self.data.train_df.loc[
            self.data.train_df[self.data.cfg.data.desired_action] == 1, ['user_id', 'photo_id']].values
        self.register_buffer("train_matrix", self._get_idx_mat(train_data), persistent=False)
        self.mgnn_weight = nn.Parameter(torch.FloatTensor([1] * self.data_info["actions_num"]))

        self.item_behaviour_W = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.emb_dim * 2, self.emb_dim * 2)) for _ in self.mgnn_weight])
        self.W = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))

        # self.item_graph_degree = trainset.item_graph_degree
        user_behaviour_degree, _ = self.__calculate_user_behaviour()

        self.register_buffer("user_behaviour_degree", user_behaviour_degree, persistent=False)
        self.message_drop = nn.Dropout(p=self.mess_dropout)
        self.train_node_drop = nn.Dropout(p=self.edge_dropout)
        self.node_drop = nn.ModuleList([nn.Dropout(p=self.edge_dropout) for _ in self.relation_dict])

        self.__param_init()

    def load_graph_data(self):
        item_graph = {}
        relation_dict = {}
        data_dir = self.data.data_dir
        for tmp_relation in self.relation:
            item_graph[tmp_relation] = torch.load(
                os.path.join(data_dir, 'item_' + tmp_relation + '.pth'))

            relation_dict[tmp_relation] = torch.load(
                os.path.join(data_dir, 'relation_' + tmp_relation + '.pth'))
        return relation_dict, item_graph

    def save_graph_data(self, relation_dict, item_graph):
        data_dir = self.data.data_dir
        for tmp_relation in self.relation:
            torch.save(relation_dict[tmp_relation], os.path.join(data_dir, 'relation_' + tmp_relation + '.pth'))
            torch.save(item_graph[tmp_relation], os.path.join(data_dir, 'item_' + tmp_relation + '.pth'))

    def __calculate_user_behaviour(self):
        for i in range(len(self.relation)):
            relation_name = self.relation[i]
            user_interaction_graph = self.get_buffer(self.relation_dict[relation_name])
            if i == 0:
                user_behaviour = torch.sparse.sum(user_interaction_graph, dim=1).to_dense().unsqueeze(-1)
                item_behaviour = torch.sparse.sum(user_interaction_graph.t(), dim=1).to_dense().unsqueeze(-1)
            else:
                user_behaviour_tmp = torch.sparse.sum(user_interaction_graph, dim=1).to_dense().unsqueeze(-1)
                item_behaviour_tmp = torch.sparse.sum(user_interaction_graph.t(), dim=1).to_dense().unsqueeze(-1)
                user_behaviour = torch.cat((user_behaviour, user_behaviour_tmp), dim=1)
                item_behaviour = torch.cat((item_behaviour, item_behaviour_tmp), dim=1)
        return user_behaviour, item_behaviour

    def __create_relation_matrix(self, df):
        '''
        create a matrix for every relation
        '''
        relation_dict = {}
        item_graph = {}
        for tmp_relation in self.relation:
            df_relation = df.loc[df[tmp_relation] == 1, ['user_id', 'photo_id']]
            df_relation = df_relation.drop_duplicates()
            index_tensor = torch.LongTensor(df_relation.values)
            lens, _ = index_tensor.shape
            relation_dict[tmp_relation] = torch.sparse_coo_tensor(index_tensor.t(),
                                                                  torch.ones(lens, dtype=torch.float),
                                                                  torch.Size([self.num_user, self.num_item]))

            user_degree = torch.sparse.sum(relation_dict[tmp_relation], dim=1).to_dense()
            D_e_user = sparse_diag(torch.nan_to_num(1 / user_degree, nan=0, posinf=0, neginf=0))
            item_graph[tmp_relation] = torch.sparse.mm(D_e_user, relation_dict[tmp_relation])

            comb_user = df_relation.groupby('user_id').apply(lambda x: list(combinations(x['photo_id'], 2)))
            index_item_tensor = torch.LongTensor(list(chain(*comb_user.values)))
            lens_item = index_item_tensor.shape[0]
            if lens_item > 0:
                item_graph[tmp_relation] = torch.sparse_coo_tensor(index_item_tensor.t(),
                                                                   torch.ones(lens_item, dtype=torch.float),
                                                                   torch.Size([self.num_item, self.num_item]))
            else:
                item_graph[tmp_relation] = torch.sparse_coo_tensor(size=torch.Size([self.num_item, self.num_item]))

            item_graph_degree = torch.sparse.sum(item_graph[tmp_relation], dim=1).to_dense()

            D_e_item = sparse_diag(torch.nan_to_num(1 / item_graph_degree, nan=0, posinf=0, neginf=0))
            item_graph[tmp_relation] = torch.sparse.mm(D_e_item, item_graph[tmp_relation])
            # item_graph_degree[tmp_relation] = torch.sparse.sum(item_graph[tmp_relation],
            #                                                    dim=1).to_dense().float().unsqueeze(-1)
        return relation_dict, item_graph

    @property
    def dataset_format(self):
        return 'SamplingTrainSet'

    def __param_init(self):
        self.item_behaviour_W = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.emb_dim * 2, self.emb_dim * 2)) for _ in self.mgnn_weight])
        for param in self.item_behaviour_W:
            nn.init.xavier_normal_(param)
        self.item_propagate_W = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim)) for _ in self.mgnn_weight])
        for param in self.item_propagate_W:
            nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.W)

    def forward(self, user, item):
        # node dropout on train matrix

        # before_time = time()
        indices = self.train_matrix._indices()
        values = self.train_matrix._values()
        values = self.train_node_drop(values)
        train_matrix = torch.sparse_coo_tensor(indices, values, size=self.train_matrix.shape)

        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)

        for i, key in enumerate(self.relation_dict):
            # node dropout
            indices = self.get_buffer(self.relation_dict[key])._indices()
            values = self.get_buffer(self.relation_dict[key])._values()
            values = self.node_drop[i](values)
            tmp_relation_matrix = torch.sparse_coo_tensor(indices, values,
                                                          size=self.get_buffer(self.relation_dict[key]).shape)

            tmp_item_propagation = torch.mm(
                torch.mm(self.get_buffer(self.item_graph[key]), self.i_embeds), self.item_propagate_W[i])
            tmp_item_propagation = torch.cat((self.i_embeds, tmp_item_propagation), dim=1)
            tmp_item_embedding = tmp_item_propagation[item]
            tmp_user_neighbour = torch.mm(tmp_relation_matrix, self.i_embeds)
            tmp_user_item_neighbour_p = torch.mm(tmp_relation_matrix, tmp_item_propagation)
            if i == 0:
                user_feature = user_behaviour_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user].expand(-1, item.shape[1], -1)
                score2 = torch.sum(tuser_tbehaviour_item_projection * tmp_item_embedding, dim=2)
            else:
                user_feature += user_behaviour_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user].expand(-1, item.shape[1], -1)
                score2 += torch.sum(tuser_tbehaviour_item_projection * tmp_item_embedding, dim=2)

        score2 = score2 / len(self.mgnn_weight)

        item_feature = torch.mm(train_matrix.t(), self.u_embeds)

        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)

        user_feature = torch.cat((self.u_embeds, user_feature), dim=1)
        item_feature = torch.cat((self.i_embeds, item_feature), dim=1)

        # message dropout
        user_feature = self.message_drop(user_feature)
        item_feature = self.message_drop(item_feature)

        tmp_user_feature = user_feature[user].expand(-1, item.shape[1], -1)
        tmp_item_feature = item_feature[item]
        score1 = torch.sum(tmp_user_feature * tmp_item_feature, dim=2)

        scores = score1 + self.lam * score2
        scores = F.softmax(scores, dim=-1)

        L2_loss = self.regularize(tmp_user_feature, tmp_item_feature)
        return scores, L2_loss

    def evaluate(self, user, item=None):
        if item is not None:
            return self.forward(user, item)

        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)

        for i, key in enumerate(self.relation_dict):

            tmp_item_propagation = torch.mm(
                torch.mm(self.get_buffer(self.item_graph[key]).float(), self.i_embeds), self.item_propagate_W[i])
            tmp_item_propagation = torch.cat((self.i_embeds, tmp_item_propagation), dim=1)

            tmp_user_neighbour = torch.mm(self.get_buffer(self.relation_dict[key]), self.i_embeds)
            tmp_user_item_neighbour_p = torch.mm(self.get_buffer(self.relation_dict[key]), tmp_item_propagation)
            if i == 0:
                user_feature = user_behaviour_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user]
                score2 = torch.mm(tuser_tbehaviour_item_projection, tmp_item_propagation.t())
            else:
                user_feature += user_behaviour_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user]
                score2 += torch.mm(tuser_tbehaviour_item_projection, tmp_item_propagation.t())

        score2 = score2 / len(self.mgnn_weight)

        item_feature = torch.mm(self.train_matrix.t(), self.u_embeds)

        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)

        user_feature = torch.cat((self.u_embeds, user_feature), dim=1)
        item_feature = torch.cat((self.i_embeds, item_feature), dim=1)

        tmp_user_feature = user_feature[user]
        score1 = torch.mm(tmp_user_feature, item_feature.t())

        scores = score1 + self.lam * score2

        return scores, 0
