from time import time
from collections import Counter, defaultdict
import torch
from torch.optim import Adam
from utility import logger
from tqdm import tqdm


class TrainManager:
    def __init__(self, cfg, data_manager, model, evaluator, loss, device):
        self.cfg = cfg
        self.data_manager = data_manager
        self.model = model
        self.evaluator = evaluator
        self.loss = loss
        self.device = device
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.lr)

        self.train_loader, self.val_loader, self.test_loader = self.data_manager.get_loaders(self.model.dataset_format)
        self.log_dir = "."
        self.Logger = logger.TrainLogger(0, max(self.cfg.epoch / 10, 20), f'ndcg@{cfg.eval.topK[-1]}', self.log_dir)

    def train(self):
        if "sampling" in self.model.dataset_format.lower():
            train_single_epoch = self.train_single_epoch_sampling
        else:
            train_single_epoch = self.train_single_epoch_all

        for epoch in range(self.cfg.epoch):
            train_loss, reg_loss_dict = train_single_epoch(self.train_loader)
            val_loss, val_quota = self.evaluator.evaluate(self.model, self.val_loader, self.loss)
            test_loss = 0
            test_quota = defaultdict(lambda: 0)
            early_stop = self.Logger.log(train_loss, reg_loss_dict, val_quota, test_quota, epoch,
                                         self.model.state_dict())

            printMetrics(epoch, train_loss, reg_loss_dict, val_loss, val_quota, self.evaluator.topK)
            if early_stop:
                logger.info('Overfitting! Early Stop at epoch %d' % epoch)
                break
        self.model.load_state_dict(self.Logger.best_weights)
        test_loss, test_quota = self.evaluator.evaluate(self.model, self.test_loader, self.loss)

        printMetrics(epoch, train_loss, reg_loss_dict, test_loss, test_quota, self.evaluator.topK)
        self.Logger.dump_log(test_quota)
        torch.cuda.empty_cache()

    def train_single_epoch_sampling(self, train_loader):
        self.model.train()
        tmp_loss = 0.
        train_reg_loss = Counter()
        num_batch = len(train_loader)
        with tqdm(train_loader) as pbar:
            for users, POIs in pbar:
                users, POIs = users.to(self.device), POIs.to(self.device)
                self.optimizer.zero_grad()
                pred, reg_loss_dict = self.model(users, POIs)
                train_reg_loss.update(reg_loss_dict)
                reg_loss = reg_loss_dict["reg_loss"]
                loss = self.loss(pred, reg_loss)
                loss.backward()
                self.optimizer.step()
                tmp_loss += loss.item()
        for key, val in train_reg_loss.items():
            train_reg_loss[key] = val / num_batch
        return tmp_loss / num_batch, train_reg_loss

    def train_single_epoch_all(self, train_loader):
        self.model.train()
        tmp_loss = 0.
        train_reg_loss = Counter()
        num_batch = len(train_loader)
        with tqdm(train_loader) as pbar:
            for users, POIs, actions, label in pbar:
                users, POIs, actions = users.to(self.device), POIs.to(self.device), actions.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                pred, reg_loss_dict = self.model(users, POIs, actions)
                train_reg_loss.update(reg_loss_dict)
                reg_loss = reg_loss_dict["reg_loss"]
                loss = self.loss(pred, reg_loss, label)
                loss.backward()
                self.optimizer.step()
                tmp_loss += loss.item()
        for key, val in train_reg_loss.items():
            train_reg_loss[key] = val / num_batch
        return tmp_loss / num_batch, train_reg_loss

    def train_single_epoch_dense(self, train_loader):
        self.model.train()
        tmp_loss = 0.
        train_reg_loss = Counter()
        num_batch = len(train_loader)
        with tqdm(train_loader) as pbar:
            for users, POIs in pbar:
                users, POIs = users.to(self.device), POIs.to(self.device)
                self.optimizer.zero_grad()
                pred, reg_loss_dict = self.model(users, POIs)
                train_reg_loss.update(reg_loss_dict)
                loss = reg_loss_dict["loss"]
                # loss = self.loss(pred, reg_loss)
                loss.backward()
                self.optimizer.step()
                tmp_loss += loss.item()
        for key, val in train_reg_loss.items():
            train_reg_loss[key] = val / num_batch
        return tmp_loss / num_batch, train_reg_loss


def printMetrics(epoch, train_loss, reg_loss_dict, test_loss, test_quota, K):
    metrics = []
    for k in K:
        metrics.append('recall@%d' % k)
        metrics.append('ndcg@%d' % k)

    reg_loss = reg_loss_dict['reg_loss']
    if 'ssl_loss' in reg_loss_dict:
        ssl_loss = reg_loss_dict['ssl_loss_scaled']
        perf_str = f'Epoch {epoch}: train==[{train_loss:.5f} = {train_loss - reg_loss:.5f} + {reg_loss - ssl_loss:.5f} + {ssl_loss:.5f}]'
    else:
        perf_str = f'Epoch {epoch}: train==[{train_loss:.5f} = {train_loss - reg_loss:.5f} + {reg_loss:.5f}]'
    if len(K) > 1:
        perf_str = perf_str + '\n'
    perf_str = perf_str + f'val==[{test_loss: .5f}], '

    for metric in metrics:
        perf_str = perf_str + metric + f'={test_quota[metric]:.5f}, '


    logger.info(perf_str)
