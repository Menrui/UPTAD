import os
import numpy as np

import torch
import torch.utils.data as data
import torchvision
from torchvision import datasets
# from torchvision import transforms
from src.datasets import transforms

from src.datasets.MVTecAD_litmodule import MVTecADDataModule
from src.datasets.YAMAHA_litmodule import YAMAHADataModule

from logging import getLogger

logger = getLogger('root')

def get_datamodule(config):
    if config.dataset.name == 'yamaha':
        return YAMAHADataModule(config=config)
    elif config.dataset.name == 'mvtec':
        return MVTecADDataModule(config=config)
    else:
        assert False, 'Incorrect dataset name.'