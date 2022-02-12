import torch
from tqdm import tqdm
from collections import defaultdict, Counter


class Evaluate:
    def __init__(self, mode, topK, device=None):
        self.topK = sorted(topK)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epison = 1e-8
        self.BIGNUM = 1e8

        dcg_list = torch.arange(2, self.topK[-1] + 2, dtype=torch.float64, device=self.device)
        self.dcg_list = torch.log2(dcg_list).pow(-1)
        self.idcg_list = torch.cumsum(self.dcg_list, dim=0)

        if mode.lower() == 'sampling':
            self.evaluate = self.evaluate_sampling
        elif mode.lower() == 'all':
            self.evaluate = self.evaluate_all
        else:
            raise ValueError('Evaluate mode not supported')

    def evaluate_all(self, model, data_loader, loss_fn=None):
        test_loss = cnt = 0
        metrics = Counter()
        with torch.no_grad():
            model.eval()
            for users, ground_truth, train_mask in tqdm(data_loader):
                users, ground_truth, train_mask = users.to(self.device), ground_truth.to(self.device), train_mask.to(
                    self.device)
                pred, _ = model.evaluate(users)
                # print(f'pred range: {pred.min().item():.4f}, {pred.max().item():.4f}')
                pred -= self.BIGNUM * train_mask
                metrics_output = self.get_metrics_all(pred, ground_truth)
                metrics.update(metrics_output[0])
                cnt += metrics_output[1]

        # print(metrics, cnt)
        for key, val in metrics.items():
            metrics[key] = val / cnt
        return test_loss, metrics

    def get_users_metrics(self, pred, ground_truth):
        is_hit = self.get_is_hit_all(pred, ground_truth)
        rdcg_list = is_hit * self.dcg_list
        num_pos = ground_truth.sum(dim=1)
        # print(f'num_pos: {num_pos[:10]}')
        # print(f'is_hit: {is_hit[:10]}')
        hit_cumsum = torch.cumsum(is_hit, dim=-1)
        metrics = defaultdict(list)

        for k in self.topK:
            ideal_hit_num = num_pos.clamp(1, k).long()
            ndcg_list = torch.sum(rdcg_list[:, :k], dim=-1) / self.idcg_list[ideal_hit_num - 1]
            recall_list = hit_cumsum[:, k - 1] / (num_pos + self.epison)
            metrics['recall@%d' % k] += recall_list.cpu().numpy().tolist()
            metrics['ndcg@%d' % k] += ndcg_list.cpu().numpy().tolist()
        return metrics, num_pos


    def get_metrics_all(self, pred, ground_truth):
        is_hit = self.get_is_hit_all(pred, ground_truth)
        rdcg_list = is_hit * self.dcg_list
        num_pos = ground_truth.sum(dim=1)
        # print(f'num_pos: {num_pos[:10]}')
        # print(f'is_hit: {is_hit[:10]}')
        hit_cumsum = torch.cumsum(is_hit, dim=-1)
        metrics = defaultdict(lambda: 0)

        for k in self.topK:
            ideal_hit_num = num_pos.clamp(1, k).long()
            ndcg_list = torch.sum(rdcg_list[:, :k], dim=-1) / self.idcg_list[ideal_hit_num - 1]
            recall_list = hit_cumsum[:, k - 1] / (num_pos + self.epison)
            metrics['recall@%d' % k] += torch.sum(recall_list).item()
            metrics['ndcg@%d' % k] += torch.sum(ndcg_list).item()

        cnt = torch.sum(num_pos > 0).item()
        return metrics, cnt

    def get_is_hit_all(self, pred, ground_truth):
        k = self.topK[-1]
        _, col_indice = torch.topk(pred, k)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            pred.shape[0], device=self.device, dtype=torch.long).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1), col_indice.view(-1)].view(-1, k)
        return is_hit

    def evaluate_sampling(self, model, data_loader, loss_fn=None):
        test_loss = cnt = 0
        num_batch = len(data_loader)
        metrics = Counter()
        with torch.no_grad():
            model.eval()
            for users, POIs in tqdm(data_loader):
                users, POIs = users.to(self.device), POIs.to(self.device)
                pred, reg_loss_dict = model.evaluate(users, POIs)
                reg_loss = reg_loss_dict["reg_loss"]
                loss = loss_fn(pred, reg_loss)
                test_loss += loss
                metrics_output = self.get_metrics_sampling(pred)
                metrics.update(metrics_output[0])
                cnt += metrics_output[1]
        for key, val in metrics.items():
            metrics[key] = val / cnt
        return test_loss / num_batch, metrics

    def get_metrics_sampling(self, pred):
        is_hit = self.get_is_hit_sampling(pred)
        rdcg_list = is_hit * self.dcg_list
        hit_cumsum = torch.cumsum(is_hit, dim=-1)
        metrics = defaultdict(lambda: 0)

        for k in self.topK:
            ndcg_list = torch.sum(rdcg_list[:, :k], dim=-1) / self.idcg_list[0]
            recall_list = hit_cumsum[:, k - 1]
            metrics['recall@%d' % k] += torch.sum(recall_list).item()
            metrics['ndcg@%d' % k] += torch.sum(ndcg_list).item()
        cnt = pred.shape[0]
        return metrics, cnt

    def get_is_hit_sampling(self, pred):
        k = self.topK[-1]
        _, col_indice = torch.topk(pred, k)
        return (col_indice == 0).double()


if __name__ == '__main__':
    ground_truth = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    pred = torch.tensor([[0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         [0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         [0.1, 0.2, 0.7, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         [0.1, 0.2, 0.3, 0.7, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         [0.1, 0.2, 0.3, 0.4, 0.7, 0.6, 0.7, 0.8, 0.9, 1.0]])
    evaluator = Evaluate('all', [1, 3, 5, 10])
    metrics, cnt = evaluator.get_metrics_all(pred, ground_truth)
    for key, val in metrics.items():
        metrics[key] = val / cnt
    print(metrics)
