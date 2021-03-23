import logging
import math
import time
import sys
sys.path.append('./')

import pickle
import torch
from thop import profile
from tqdm import tqdm
import numpy as np

from models.mlp import MLP
from models.supernet import MultiResolutionNet
from utils.config import parse_args
from utils.logger_util import setup_logging
from utils.train_utils import get_criterion, get_dataloader
from utils.utils import set_env


def _make_divisible(v, divisor, min_value=None):
    """Make channels divisible to divisor.

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return int(new_v)


def net2flops(embedding, device):
    input_channels = embedding[0:2]
    block_1 = embedding[2:3] + [1] + embedding[3:5]
    block_2 = embedding[5:6] + [2] + embedding[6:10]
    block_3 = embedding[10:11] + [3] + embedding[11:17]
    block_4 = embedding[17:18] + [4] + embedding[18:26]
    last_channel = embedding[26]
    network_setting = []
    for item in [block_1, block_2, block_3, block_4]:
        for _ in range(item[0]):
            network_setting.append([item[1], item[2:-int(len(item) / 2 - 1)], item[-int(len(item) / 2 - 1):]])
    model = MultiResolutionNet(input_channel=input_channels,
                              network_setting=network_setting,
                              last_channel=last_channel).to(device)
    # Get Flops and Parameters
    input = torch.randn(1, 3, 128, 128).to(device)
    macs, params = profile(model, inputs=(input,), verbose=False)
    return macs / 1e9, params / 1e6


def normalize(data, cfg, device):
    normalized_data = (data - cfg.data.input_mean) / cfg.data.input_std
    normalized_data = normalized_data.float().to(device)
    normalized_data.requires_grad = True
    return normalized_data


def denormalize(normalized_data, cfg):
    normalized_data = normalized_data[0].detach().cpu().numpy().tolist()
    data = cfg.data.input_mean + normalized_data * cfg.data.input_std
    return data


def predicted_metrics(model, normalized_data, cfg):
    prediction = model(normalized_data.detach())[0].detach().cpu().numpy().tolist()
    return prediction * cfg.data.output_std + cfg.data.output_mean


def main(cfg):
    device = set_env(cfg)

    logging.info('Loading the dataset.')
    _, criterion = get_criterion(cfg.optimization.criterion)
    train_dataloader, val_dataloader = get_dataloader(cfg)

    model = MLP(**cfg.network).to(device)
    model.load_state_dict(torch.load(f'{cfg.editing.model_path}/{cfg.data.dataset}/best_model.pth'))
    model.eval()
    logging.info(f'Constructing model on the {device}:{cfg.CUDA_DEVICE}.')
    logging.info(model)

    cfg.data.max_value = torch.tensor([64, 64,
                                       2, 2, 64,
                                       2, 2, 2, 64, 64,
                                       2, 2, 2, 2, 64, 64, 64,
                                       2, 2, 2, 2, 2, 64, 64, 64, 64,
                                       64]) / 2 * cfg.editing.max_value_alpha
    cfg.data.min_value = torch.tensor([16, 16,
                                       2, 2, 16,
                                       2, 2, 2, 16, 16,
                                       2, 2, 2, 2, 16, 16, 16,
                                       2, 2, 2, 2, 2, 16, 16, 16, 16,
                                       16]) / 2
    cfg.data.normalized_max_value = (cfg.data.max_value - cfg.data.input_mean) / cfg.data.input_std
    cfg.data.normalized_min_value = (cfg.data.min_value - cfg.data.input_mean) / cfg.data.input_std

    data = torch.tensor([[64, 64,
                          2, 2, 64,
                          2, 2, 2, 64, 64,
                          2, 2, 2, 2, 64, 64, 64,
                          2, 2, 2, 2, 2, 64, 64, 64, 64,
                          64]])

    # data = torch.tensor([[40, 40, 1, 4, 8, 1, 1, 2, 104, 64, 1, 3, 1, 1, 56, 32, 88, 3, 1, 1, 3, 3, 8, 64, 128, 128, 16]])  # seg
    # data = torch.tensor([[88, 128, 1, 1, 128, 1, 1, 4, 120, 32, 2, 1, 2, 4, 128, 128, 128, 1, 1, 1, 1, 1, 32, 128, 8, 8, 128]])  # cls
    # data = torch.tensor([[8, 8, 1, 1, 88, 1, 1, 1, 8, 8, 1, 1, 1, 1, 64, 128, 128, 1, 1, 1, 1, 4, 80, 128, 128, 48, 128]])  # video
    # data = torch.tensor([[120, 48, 1, 1, 24, 1, 3, 4, 80, 128, 1, 1, 3, 1, 96, 8, 128, 1, 1, 1, 2, 1, 40, 80, 40, 96, 112]])  # 3ddet


    normalized_data = normalize(data, cfg, device)
    denormalized_data = data[0].numpy()
    rounded_data = denormalized_data.copy()

    # original_metrics = predicted_metrics(model, normalized_data, cfg)

    flops, params = net2flops(data[0].int().cpu().numpy().tolist(), device)

    edit_net_set = list()

    for iter in tqdm(range(cfg.editing.iters)):
        optimizer = torch.optim.SGD([normalized_data], lr=cfg.editing.lr)
        optimizer.zero_grad()
        model.zero_grad()

        pred = model(normalized_data)[0]
        main_record = model(normalize(torch.Tensor(rounded_data).unsqueeze(0), cfg, device))[0][0]
        main_metric = pred[0]
        main_metric_target = main_metric.clone().detach() + cfg.editing.per_step_increase
        loss = criterion(main_metric, main_metric_target)

        net_dict = {
            'rounded_net': rounded_data,
            'continuous_net': denormalized_data,
            'predicted_metrics': pred.detach().cpu().numpy().tolist() * cfg.data.output_std + cfg.data.output_mean,
            'main_metric': main_record.detach().cpu().item() * cfg.data.output_std[0] + cfg.data.output_mean[0],
            'flops': flops,
            'params': params
        }
        edit_net_set.append(net_dict)
        print(net_dict)

        if cfg.editing.use_flops:
            flops = pred[-2]
            flops_target = flops.clone().detach() - cfg.editing.per_flops_decrease
            loss = loss + cfg.editing.alpha * criterion(flops, flops_target)

        loss.backward()
        optimizer.step()

        for i in range(normalized_data.shape[1]):
            if normalized_data[0][i] > cfg.data.normalized_max_value[i]:
                normalized_data[0][i] = cfg.data.normalized_max_value[i]
            if normalized_data[0][i] < cfg.data.normalized_min_value[i]:
                normalized_data[0][i] = cfg.data.normalized_min_value[i]
        normalized_data = normalized_data.detach().clone()
        normalized_data.requires_grad = True

        denormalized_data = denormalize(normalized_data, cfg)
        rounded_data = denormalized_data.copy()
        for i in range(rounded_data.shape[0]):
            rounded_data[i] = _make_divisible(rounded_data[i], cfg.data.min_value[i])
        flops, params = net2flops(list(rounded_data.astype(int)), device)

    pickle.dump(edit_net_set, open(f"{cfg.log_dir}/NCP.pkl", "wb"))


if __name__ == '__main__':
    cfg = parse_args()
    # alias = f'epoch_{cfg.optimization.epoch}-bs_{cfg.optimization.batch_size}' \
    #         f'-{cfg.optimization.optimizer}-{cfg.optimization.scheduler}'
    # cfg.timestamp = time.strftime('{}-%Y%m%d-%H%M%S-{}'.format(cfg.data.dataset, alias))
    cfg.log_dir = '{}/{}'.format(cfg.log_dir, cfg.data.dataset)
    setup_logging(cfg.log_dir, file_name='NCP.log')
    logging.info(cfg)

    main(cfg)
