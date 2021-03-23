import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def _get_env(env_name):
    if env_name not in os.environ:
        raise RuntimeError('${} should be set'.format(env_name))
    return os.environ[env_name]


def init_dist(backend='nccl', **kwargs):
    rank = int(_get_env('RANK'))
    local_rank = int(_get_env('LOCAL_RANK'))  # GPU
    assert rank % torch.cuda.device_count() == local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)
    assert dist.is_initialized()


def set_env(cfg):
    logging.info('Setting environments.')

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.cuda.set_device(cfg.CUDA_DEVICE)
        torch.cuda.manual_seed_all(cfg.SEED)
    else:
        device = 'cpu'

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    cfg.device = device
    return device
