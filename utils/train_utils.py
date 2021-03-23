import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from datasets.dataset import BaseDataset, DataStat


def get_criterion(name):
    if name == 'mse':
        train_criterion = nn.MSELoss()
    elif name == 'l1':
        train_criterion = nn.L1Loss()
    elif name == 'smoothl1':
        train_criterion = nn.SmoothL1Loss()

    val_criterion = nn.L1Loss(reduction='none')

    return train_criterion, val_criterion


def get_optimization(cfg, model):
    if cfg.optimization.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     **cfg.optimization.adam_param)
        cfg.optimization.onecycle_scheduler.max_lr = cfg.optimization.adam_param.lr
    elif cfg.optimization.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      **cfg.optimization.adam_param)
        cfg.optimization.onecycle_scheduler.max_lr = cfg.optimization.adam_param.lr
    elif cfg.optimization.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    **cfg.optimization.sgd_param)
        cfg.optimization.onecycle_scheduler.max_lr = cfg.optimization.sgd_param.lr

    if cfg.optimization.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer,
                                               **cfg.optimization.exp_scheduler)
    elif cfg.optimization.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             **cfg.optimization.step_scheduler)
    elif cfg.optimization.scheduler == 'onecycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                            **cfg.optimization.onecycle_scheduler)
    elif cfg.optimization.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   **cfg.optimization.cosine_scheduler)
    return optimizer, scheduler


def get_dataloader(cfg):
    data_stat = DataStat(cfg, cfg.data.dataset)

    train_dataset = BaseDataset(data_stat, mode='train')
    test_dataset = BaseDataset(data_stat, mode='val')

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg.optimization.batch_size,
                                  shuffle=True,
                                  num_workers=4 if cfg.device == 'gpu' else 0)

    val_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=cfg.optimization.batch_size,
                                shuffle=False,
                                num_workers=4 if cfg.device == 'gpu' else 0)

    return train_dataloader, val_dataloader
