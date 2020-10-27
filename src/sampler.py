import numpy as np
import torch


def get_sampler(config):
    if config.model.sampler == 'normal':
        sampler = torch.randn
    elif config.model.sampler == 'uniform':
        sampler = torch.rand
    return sampler
