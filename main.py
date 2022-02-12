from omegaconf import DictConfig
import hydra
import os
import numpy as np
import torch

import utility

from utility import BPRLoss, LogLoss
from utility import logger
from model import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)[1:-1]

    setup_seed(cfg.seed)
    logger.info("Start experiment!")

    logger.info("Loading data...")
    data_manager = utility.DataManager(cfg)
    logger.info(f"Data information {data_manager.data_info}")

    model = eval(cfg.model.name)
    logger.info(f"Creating Model {cfg.model.name}...")
    model = model(cfg.model, data_manager)

    loss = eval(cfg.loss)
    loss = loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluator = utility.Evaluate(cfg.eval.mode, cfg.eval.topK, device)
    train_manager = utility.TrainManager(cfg, data_manager, model, evaluator, loss, device)
    logger.info("Start training...")
    train_manager.train()


if __name__ == "__main__":
    my_app()
