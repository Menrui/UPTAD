import os
from typing import Any, Optional, Union, List

import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision

from pytorch_lightning import LightningDataModule

from src.datasets import transforms
from src.datasets.MVTecAD import MVTecAD

from logging import getLogger

logger = getLogger('root')


class MVTecADDataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()

        self.data_dir = os.path.join(config.work_dir, '{}/mvtec'.format(config.dataset.data_dir))
        self.category = config.dataset.category
        self.data_config = config.dataset
        self.mask_config = config.model.mask

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            transform = torchvision.transforms.Compose([
                # transforms.Resize((self.data_config.img_h, self.data_config.img_w)),
                # transforms.RandomCrop((self.data_config.patch_h, self.data_config.patch_w)),
                transforms.RandomFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            dataset = MVTecAD(
                root=self.data_dir, category=self.category, train=True, transform=transform,
                size_data=(self.data_config.img_h, self.data_config.img_w, self.data_config.img_nch),
                maskconf=self.mask_config
            )
            n_samples = len(dataset)
            train_size = int(n_samples * (1-0.036))
            val_size = n_samples - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            logger.info('train image shape: {}'.format(self.train_dataset[0][0].shape))

        if stage == 'test' or stage is None:
            transform = torchvision.transforms.Compose([
                # transforms.Resize((self.data_config.img_h, self.data_config.img_w)),
                # transforms.UnifromSample((self.data_config.patch_h, self.data_config.patch_w)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            self.test_dataset = MVTecAD(
                root=self.data_dir, category=self.category, train=False, transform=transform,
                size_data=(self.data_config.img_h, self.data_config.img_w, self.data_config.img_nch),
                maskconf=self.mask_config
            )
            logger.info('test image shape: {}'.format(self.test_dataset[0][0].shape))

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            shuffle=False,
            drop_last=False
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.data_config.test_batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            shuffle=False,
            drop_last=False
        )
