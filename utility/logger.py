import copy
import os
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch

import logging

info = logging.info


class TrainLogger:
    def __init__(self, threshold, maxdown, benchmark, log_dir):
        """
        :param threshold: metrics / best_metrics > 1 + threshold, set the early stop flag true
        :param maxdown: Maximum number of epochs allowed where the metrics is going down
        :param benchmark: `p` | `map` | `ndcg` | `mrr` | `hit` | `r` | `f`
        """
        self.threshold = threshold
        self.maxdown = maxdown
        self.benchmark = benchmark
        self.best_metrics = defaultdict(lambda: 0)
        self.test_metrics = None
        self.best_weights = None
        self.best_epoch = -1
        self.down = 0
        self.last_metric = defaultdict(lambda: 0)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def log(self, loss, reg_loss, metrics, test_metrics, epoch, state_dict):
        self.writer.add_scalar('loss/total_loss', loss, epoch)

        for key, values in reg_loss.items():
            self.writer.add_scalar('loss/' + key, values, epoch)
        self.writer.add_scalar('loss/bpr_loss', loss - reg_loss["reg_loss"], epoch)

        for k, v in metrics.items():
            self.writer.add_scalar("val_metrics/" + k, v, epoch)
        for k, v in test_metrics.items():
            self.writer.add_scalar("test_metrics/" + k, v, epoch)

        if self.best_metrics[self.benchmark] > 0 and \
                metrics[self.benchmark] / self.best_metrics[self.benchmark] < self.threshold:
            return True
        if 0 < self.best_metrics[self.benchmark] < metrics[self.benchmark]:
            self.down = 0
        else:
            self.down += 1
        self.last_metric = metrics
        if metrics[self.benchmark] > self.best_metrics[self.benchmark]:
            self.best_metrics = copy.deepcopy(metrics)
            self.test_metrics = copy.deepcopy(test_metrics)
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(state_dict)
        return self.down > self.maxdown

    def dump_log(self, test_metrics):
        metrics = self.best_metrics
        test_metric = {'test_' + k: v for k, v in test_metrics.items()}
        metrics.update(test_metric)
        pth_name = f"model.pt"
        torch.save(self.best_weights, os.path.join(self.log_dir, pth_name))
        df = pd.DataFrame(metrics, index=[0])
        self.info(f"Best Metrics\n{df}")
        df.to_csv(os.path.join(self.log_dir, "metrics.csv"), index=False)

    @staticmethod
    def info(info_str):
        logging.info(info_str)
