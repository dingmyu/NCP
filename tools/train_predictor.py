import logging
import math
import time
import sys
sys.path.append('./')

import pickle
import torch

from models.mlp import MLP
from utils.config import parse_args
from utils.logger_util import setup_logging
from utils.train_utils import get_criterion, get_optimization, get_dataloader
from utils.utils import set_env


def train_epoch(model,
                dataloader,
                criterion,
                optimizer,
                scheduler,
                device,
                epoch,
                cfg):
    model.train()
    total_loss = 0.0

    for i, [data, label] in enumerate(dataloader):
        pred = model(data.to(device))

        loss = criterion(pred, label.to(device))

        model.zero_grad()
        loss.backward()
        optimizer.step()
        if cfg.optimization.scheduler in ['cosine', 'onecycle']:
            scheduler.step()

        total_loss += loss.item()
        if i % cfg.log_interval == 0 and i > 0:
            cur_loss = total_loss / cfg.log_interval
            logging.info(
                '| train | epoch {:3d} | {:4d}/{:4d} batches | lr {:5.5f} | '
                'loss {:5.5f} | ppl {:5.5f}'.format(epoch,
                                                    i,
                                                    len(dataloader),
                                                    scheduler.get_lr()[0],
                                                    cur_loss,
                                                    math.exp(cur_loss)))
            total_loss = 0.0


def valid_epoch(model, dataloader, criterion, device, epoch, cfg):
    model.eval()
    total_loss = torch.zeros(len(cfg.data.combined_metrics)).to(device)

    with torch.no_grad():
        for _, [data, label] in enumerate(dataloader):

            pred = model(data.to(device))
            label = label.to(device)

            std = torch.from_numpy(cfg.data.output_std).to(device)
            mean = torch.from_numpy(cfg.data.output_mean).to(device)
            pred = pred * std + mean
            label = label * std + mean

            loss = criterion(pred, label).mean(0)
            total_loss += loss

        cur_loss = total_loss / len(dataloader)

        print_info = '| valid | epoch {:3d} | Avg loss {:5.5f}'.format(
            epoch, cur_loss.mean().item())
        for index, metric in enumerate(cfg.data.combined_metrics):
            print_info += '| {}_loss {:5.5f}'.format(metric, cur_loss[index])
        logging.info(print_info)
    return cur_loss[0].item()  # cur_loss.mean().item()


def main(cfg):
    device = set_env(cfg)

    logging.info('Loading the dataset.')
    train_criterion, val_criterion = get_criterion(cfg.optimization.criterion)
    train_dataloader, val_dataloader = get_dataloader(cfg)

    model = MLP(**cfg.network).to(device)
    logging.info(f'Constructing model on the {device}:{cfg.CUDA_DEVICE}.')
    logging.info(model)

    # Set total steps for onecycleLR and cosineLR
    cfg.optimization.total_steps = len(train_dataloader) * cfg.optimization.epoch
    cfg.optimization.onecycle_scheduler.total_steps = \
        cfg.optimization.cosine_scheduler.T_max = cfg.optimization.total_steps

    optimizer, scheduler = get_optimization(cfg, model)

    best_loss = float("inf")
    for epoch in range(cfg.optimization.epoch):
        train_epoch(model,
                    train_dataloader,
                    train_criterion,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                    cfg)
        if cfg.optimization.scheduler in ['exp', 'step']:
            scheduler.step()

        val_loss = valid_epoch(model,
                               val_dataloader,
                               val_criterion,
                               device,
                               epoch,
                               cfg)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '{}/best_model.pth'.format(cfg.log_dir))
            logging.info(f'New Main Loss {best_loss}')
    torch.save(model.state_dict(), '{}/last_model.pth'.format(cfg.log_dir))

    logging.info('Best Main Loss {}'.format(best_loss))
    logging.info(cfg)
    pickle.dump(cfg, open('{}/config.pkl'.format(cfg.log_dir), 'wb'))


if __name__ == '__main__':
    cfg = parse_args()
    # alias = f'epoch_{cfg.optimization.epoch}-bs_{cfg.optimization.batch_size}' \
    #         f'-{cfg.optimization.optimizer}-{cfg.optimization.scheduler}'
    # cfg.timestamp = time.strftime('{}-%Y%m%d-%H%M%S-{}'.format(cfg.data.dataset, alias))
    cfg.log_dir = '{}/{}'.format(cfg.log_dir, cfg.data.dataset)
    setup_logging(cfg.log_dir)
    logging.info(cfg)

    main(cfg)
