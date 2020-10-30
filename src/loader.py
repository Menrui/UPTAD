import os
import numpy as np

import torch
import torch.utils.data as data
import torchvision
from torchvision import datasets
# from torchvision import transforms
from src.datasets import transforms

from src.datasets.MVTecAD import MVTecAD
from src.datasets.mnist import MNIST
from src.datasets.YAMAHA import YAMAHA

from logging import getLogger

logger = getLogger('root')


def get_loader(config, is_train=True):
    transform = get_transform(config, is_train)
    dataset = get_dataset(config, is_train, transform)

    if is_train:
        batch_size = config.dataset.batch_size
    else:
        batch_size = 1
    if is_train:
        n_samples = len(dataset)
        train_size = int(len(dataset) * 0.8)
        val_size = n_samples - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
            shuffle=is_train,
            drop_last=is_train,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, val_loader
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
            shuffle=is_train,
            drop_last=is_train,
        )
        return loader


def get_transform(config, is_train):
    transform = []
    if config.dataset.name == 'mnist' or config.dataset.name == 'kmnist' or config.dataset.name == 'fmnist':
        transform = [
            transforms.Resize((config.dataset.img_h, config.dataset.img_w)),
            # transforms.RandomCrop(config.dataset.patch_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    if config.dataset.name == 'mvtec' and is_train is True:
        transform = [
            transforms.Resize((config.dataset.img_h, config.dataset.img_w)),
            transforms.RandomCrop(config.dataset.patch_size),
            transforms.RandomFlip(),
            # transforms.RandomHorizontalFlip(),
            # RandomFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if config.dataset.img_nch==3 else transforms.Normalize([0.5], [0.5])
        ]
    elif config.dataset.name == 'mvtec' and is_train is False:
        transform = [
            transforms.Resize((config.dataset.img_h, config.dataset.img_w)),
            transforms.UnifromSample(config.dataset.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if config.dataset.img_nch==3 else transforms.Normalize([0.5], [0.5])
        ]
    if config.dataset.name == 'yamaha' and is_train is True:
        transform = [
            transforms.RandomCrop(config.dataset.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if config.dataset.img_nch==3 else transforms.Normalize([0.5], [0.5])
        ]
    elif config.dataset.name == 'yamaha' and is_train is False:
        transform = [
            transforms.UnifromSample(config.dataset.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) if config.dataset.img_nch==3 else transforms.Normalize([0.5], [0.5])
        ]
    return torchvision.transforms.Compose(transform)


def get_dataset(config, is_train, transform):
    if config.dataset.name == 'mnist':
        normal_digit = config.dataset.digit if is_train else None
        dataset = MNIST(
            root=os.path.join(config.work_dir, config.dataset.data_dir),
            normal_digit=normal_digit, train=is_train,
            download=True, transform=transform
        )
    elif config.dataset.name == 'mvtec':
        dataset = MVTecAD(
            root=os.path.join(config.work_dir, '{}/mvtec'.format(config.dataset.data_dir)),
            category=config.dataset.category, train=is_train, transform=transform,
            size_data=(config.dataset.img_h, config.dataset.img_w, config.dataset.img_nch),
            maskconf=config.dataset.mask
        )
    elif config.dataset.name == 'yamaha':
        dataset = YAMAHA(
            root=os.path.join(config.work_dir, '{}/yamaha'.format(config.dataset.data_dir)),
            category=config.dataset.category, train=is_train, transform=transform,
            size_data=(config.dataset.img_h, config.dataset.img_w, config.dataset.img_nch),
            maskconf=config.dataset.mask
        )
    else:
        raise ValueError('Invalid dataset: {}'.format(config.dataset))

    image, _, _, _, _, _ = dataset[0]
    config.dataset.data_info = np.concatenate([image.shape]).tolist() if image.ndim == 3 else np.concatenate([image.shape]).tolist()[1:]
    logger.info('image shape: {}'.format(np.concatenate([image.shape])))

    return dataset


class RandomFlip(object):
    def __call__(self, input):
        if np.random.rand() > 0.5:
            input = np.fliplr(input)
        if np.random.rand() > 0.5:
            input = np.flipud(input)
        return input
