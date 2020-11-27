import os
import sys
import argparse

import hydra
import random
import numpy as np
import torch

import pytorch_lightning as pl

from hydra.experimental import initialize, compose
from PIL import Image

from src.data_module import get_datamodule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--ch', action='store_false')
    args = parser.parse_args()

    print(args.output_dir)
    initialize(config_path=f'{args.output_dir}/config')
    config = compose(config_name='config.yaml')
    config.work_dir = os.getcwd()

    config.dataset.patch_w = 256
    config.dataset.patch_h = 256
    config.model.mask.mask_mode = 'chole'
    config.model.mask.mask_ratio = 0.85
    config.model.mask.mask_w = 3
    config.model.mask.mask_h = 3


    data_module = get_datamodule(config=config)
    data_module.setup(stage='fit')
    loader = data_module.train_dataloader()

    for i, batch in enumerate(loader):
        img, input, label, mask, _, target = batch
        if i % 10 == 0:
            Image.fromarray(((img.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] * 0.5 + 0.5)*255).astype(np.uint8)).save(f'./sample/original_{i}.png')
            Image.fromarray(((input.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] * 0.5 + 0.5)*255).astype(np.uint8)).save(f'./sample/{os.path.basename(args.output_dir)}input_{i}.png')
            Image.fromarray(((mask.detach().cpu().numpy().transpose(0, 2, 3, 1)[0])*255).astype(np.uint8)).save(f'./sample/{os.path.basename(args.output_dir)}mask_{config.model.mask.mask_mode}_{i}.png')
            print(i)

