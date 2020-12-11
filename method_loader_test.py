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

    categorys = ['cir', 'line', 'rec']
    sigmas = [1,3,5]
    mask_ratios = [0.85, 0.75, 0.65]

    for category in categorys:
        for sigma in sigmas:
            for ratio in mask_ratios:


                config.model.mask.category=category
                config.model.mask.size=10 
                config.model.mask.mu=0 
                config.model.mask.sigma=sigma
                config.model.mask.mask_ratio=ratio
                config.model.mask.color='c'

                data_module = get_datamodule(config=config)
                data_module.setup(stage='fit')
                loader = data_module.val_dataloader()

                for i, batch in enumerate(loader):
                    img, input, label, mask, _, target = batch
                    if i % 10 == 0:
                        Image.fromarray(((img.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] * 0.5 + 0.5)*255).astype(np.uint8)).save(f'./sample/original_{i}.png')
                        Image.fromarray(((input.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] * 0.5 + 0.5)*255).astype(np.uint8)).save(f'./sample/{os.path.basename(args.output_dir)}input_{config.model.mask.category}_sigma{config.model.mask.sigma}_ratio{config.model.mask.mask_ratio}_{i}.png')
                        Image.fromarray(((mask.detach().cpu().numpy().transpose(0, 2, 3, 1)[0])*255).astype(np.uint8)).save(f'./sample/{os.path.basename(args.output_dir)}mask_{config.model.mask.category}_sigma{config.model.mask.sigma}_ratio{config.model.mask.mask_ratio}_{i}.png')
                        print(i)
                    break

