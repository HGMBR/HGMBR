import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, data_info, data_cfg, *args, **kwargs):
        super().__init__()
        self.data_info = data_info
        self.data_cfg = data_cfg

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _get_idx_mat(self, train_data):
        value_fill_mat = np.ones(train_data.shape[0])
        shape = (self.data_info['user_ids_num'], self.data_info['photo_ids_num'])
        train_mat = sp.coo_matrix((value_fill_mat, (train_data[:, 0], train_data[:, 1])),
                                  shape=shape, dtype=np.float32)
        train_mat = train_mat.todok()
        return train_mat


class SamplingTrainSet(Dataset):
    def __init__(self, data_info, data_cfg, df):
        super().__init__(data_info, data_cfg, df)
        self.num_item = self.data_info['photo_ids_num']
        self.num_ng = self.data_cfg['num_ng']
        self.train_data = df.loc[df[data_cfg.desired_action] == 1, ['user_id', 'photo_id']].values
        self.train_mat = self._get_idx_mat(self.train_data)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        user, pos_item = self.train_data[idx]
        all_item = [pos_item]
        while len(all_item) <= self.num_ng:
            j = np.random.randint(self.num_item)
            while (user, j) in self.train_mat or j in all_item:
                j = np.random.randint(self.num_item)
            all_item.append(j)
        return torch.LongTensor([user]), torch.LongTensor(all_item)


class AllTrainSet(Dataset):
    def __init__(self, data_info, data_cfg, df):
        super().__init__(data_info, data_cfg, df)
        self.train_data = df.values

    def _get_mbr_data(self, df):
        df_all = []
        for i, action in enumerate(self.data_cfg.effective_columns):
            df_tmp = df[['user_id', 'photo_id', action]].copy()
            df_tmp["action"] = i
            df_tmp = df_tmp.rename(columns={action: 'label'})
            df_all.append(df_tmp)
        df = pd.concat(df_all)
        return df

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        entry = self.train_data[idx]
        return torch.LongTensor([entry[0]]), torch.LongTensor([entry[1]]), torch.LongTensor(
            [entry[2]]), torch.FloatTensor([entry[3]])


class SamplingTestSet(Dataset):
    def __init__(self, data_info, data_cfg, eval_cfg, df_train, df_test, seed=123):
        super().__init__(data_info, data_cfg, df_train)
        self.data_cfg = data_cfg
        self.eval_cfg = eval_cfg
        self.num_item = self.data_info['photo_ids_num']
        self.num_ng = self.eval_cfg.num_negative_eval

        if len(df_train.columns) > 2:
            train_data = df_train.loc[df_train[data_cfg.desired_action] == 1, ['user_id', 'photo_id']].values
        else:
            train_data = df_train.values
        self.train_mat = self._get_idx_mat(train_data)

        if len(df_test.columns) == 2:
            self.test_data = df_test[['user_id', 'photo_id']].values
        else:
            self.test_data = df_test.loc[df_test[data_cfg.desired_action] == 1, ['user_id', 'photo_id']].values
        np.random.seed(seed)
        self._ng_sample()

    def _ng_sample(self):
        self.users = []
        self.items = []
        for x in self.test_data:
            u, i = x[0], x[1]
            all_item = {i: 0}
            while len(all_item) <= self.num_ng:
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                all_item[j] = 0
            all_item = list(all_item.keys())
            self.users.append([u])
            self.items.append(all_item)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        return torch.LongTensor(user), torch.LongTensor(item)


class AllTestSet(Dataset):
    def __init__(self, data_info, data_cfg, eval_cfg, df_train, df_test, seed=123):
        super().__init__(data_info, data_cfg, df_train)
        if len(df_train.columns) > 2:
            train_data = df_train.loc[df_train[data_cfg.desired_action] == 1, ['user_id', 'photo_id']].values
        else:
            train_data = df_train.values
        test_data = df_test.values

        self.train_mask = self._get_idx_mat(train_data).tocsr()
        self.ground_truth = self._get_idx_mat(test_data).tocsr()

    def __getitem__(self, index):
        return index, torch.from_numpy(self.ground_truth[index].toarray()).squeeze(), torch.from_numpy(
            self.train_mask[index].toarray()).float().squeeze()

    def __len__(self):
        return self.data_info['user_ids_num']


__all__ = [s for s in dir() if "trainset" in s.lower() or "testset" in s.lower()]
