import os
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
import pandas as pd
from utility.dataset import *
from utility import logger


class DataManager:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        if cfg.data.data_dir.startswith("~"):
            self.data_dir = os.path.expanduser(cfg.data.data_dir)
        else:
            self.data_dir = to_absolute_path(cfg.data.data_dir)
        self.train_df, self.val_df, self.test_df, self.data_info = self.load_data()

    def load_data(self):
        logger.info(f'Loading data from {self.data_dir}...')
        effective_columns = self.cfg.data.effective_columns
        test_columns = self.cfg.data.test_columns
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"), usecols=effective_columns)[effective_columns]
        val_df = pd.read_csv(os.path.join(self.data_dir, "val.csv"), usecols=test_columns)[test_columns]
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"), usecols=test_columns)[test_columns]
        data_info = dict()
        with open(os.path.join(self.data_dir, "size.txt"), "r") as f:
            data_info['user_ids_num'] = int(f.readline())
            data_info['photo_ids_num'] = int(f.readline())

        data_info["actions_num"] = len(train_df.columns) - 2
        data_info["cols"] = list(train_df.columns)

        target_behavior_num = len(train_df.loc[train_df[self.cfg.data.desired_action] == 1])
        actions_list = effective_columns[2:]
        total_behavior_num = (train_df[actions_list] != 0).sum(1).sum()

        denominator = data_info['user_ids_num'] * data_info['photo_ids_num']
        logger.info(f'Data loaded.')
        logger.info(f'Target behavior num: {target_behavior_num}, density: {target_behavior_num / denominator:.5f}')
        logger.info(f'Total behavior num: {total_behavior_num}, density: {total_behavior_num / denominator:.5f}')
        return train_df, val_df, test_df, data_info

    def get_loaders(self, trainset_format='SamplingTrainSet', shuffle=True):
        trainset = eval(trainset_format)
        data_cfg = self.cfg.data
        eval_cfg = self.cfg.eval
        num_workers = self.cfg.workers
        train_set = trainset(self.data_info, data_cfg, self.train_df)
        train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=shuffle, num_workers=num_workers)

        testset_format = eval_cfg.testset_format
        testset = eval(testset_format)
        train_desired = self.train_df.loc[self.train_df[data_cfg.desired_action] == 1, ['user_id', 'photo_id']]
        if len(self.val_df.columns) > 2:
            val_desired = self.val_df.loc[self.val_df[data_cfg.desired_action] == 1, ['user_id', 'photo_id']]
            test_desired = self.test_df.loc[self.test_df[data_cfg.desired_action] == 1, ['user_id', 'photo_id']]
        else:
            val_desired = self.val_df
            test_desired = self.test_df

        val_train_mask = pd.concat([train_desired, test_desired])
        val_set = testset(self.data_info, data_cfg, eval_cfg, val_train_mask, self.val_df, self.cfg.seed + 1)
        val_loader = DataLoader(val_set, batch_size=eval_cfg.batch_size_eval, shuffle=False, num_workers=num_workers)

        test_train_mask = pd.concat([train_desired, val_desired])
        test_set = testset(self.data_info, data_cfg, eval_cfg, test_train_mask, self.test_df, self.cfg.seed + 2)
        test_loader = DataLoader(test_set, batch_size=eval_cfg.batch_size_eval, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader
