import os
import numpy as np
from PIL import Image
import cv2
import glob
import copy

import torch
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
import matplotlib.pyplot as plt
import numba

from src.datasets.argument import NoiseGrinder

class MVTecAD(torch.utils.data.Dataset):

    def __init__(self, root, category, train: bool, transform=None, size_data=(256, 256, 3), 
                #  ratio=0.97, mask_size=(2, 2), mask_overcoat=True, mask_mode='normal', add_gauss=False):
                ):
        """
        :param root:        MVTecAD dataset dir
        :param category:    MVTecAD category
        :param train:       If it is true, the training mode
        :param transform:   pre-processing
        :param sgm:         noise parameter
        :param ratio:       noise ratio
        :param size_data:   dataset image size
        :param mask_size:   mask_size[0] is the width of the mask (integer or tuple). mask_size[1] is the height of the mask (integer or tuple).
        """
        self.root = root
        self.category = category
        self.train = train
        self.transform = transform

        self.noise_grinder = NoiseGrinder(maskconf, size_data=size_data, noise_sgm=sgm)

        self.sgm = sgm
        self.size_data = size_data
        self.add_loss_mask = maskconf.add_loss_mask
        self.ratio = maskconf.mask_ratio
        self.mask_mode = maskconf.mask_mode
        self.mask_fill = maskconf.mask_fill
        self.mask_size = (maskconf.mask_h, maskconf.mask_w)
        self.mask_overcoat = maskconf.mask_overcoat
        self.add_gauss = maskconf.add_gauss

        self.train_dir = os.path.join(root, category, 'train')
        self.test_dir = os.path.join(root, category, 'test')

        self.normal_class = ['good']
        self.abnormal_class = os.listdir(self.test_dir)
        self.abnormal_class.remove(self.normal_class[0])

        if self.train:
            img_paths = glob.glob(os.path.join(
                self.train_dir, self.normal_class[0], '*.png'
            ))
            self.img_paths = sorted(img_paths)
            self.labels = len(self.img_paths) * [1]
        else:
            img_paths = []
            labels = []
            gt_paths = []
            for c in os.listdir(self.test_dir):
                # if "liquid" in c:
                #     continue
                paths = glob.glob(os.path.join(
                    self.test_dir, c, '*.png'
                ))
                img_paths.extend(sorted(paths))
                gt_paths.extend(sorted(glob.glob(os.path.join(self.gt_dir, c, '*.png'))))
                if c == self.normal_class[0]:
                    labels.extend(len(paths) * [0])
                else:
                    for i,abclass in enumerate(self.abnormal_class):
                        if c == abclass:
                            labels.extend(len(paths) * [i+1])

            self.img_paths = img_paths
            self.labels = labels


    def __getitem__(self, index):
        """
        :return:
            original:    original image
            input:  input data to the model
            label:  original image + noise
            mask:   blind spot index
        """
        img_path, target = self.img_paths[index], self.labels[index]

        original = plt.imread(img_path)

        original = np.expand_dims(original, axis=2) if original.ndim==2 else original

        if self.transform:
            original = self.transform(Image.fromarray((original*255).astype(np.uint8).squeeze()))

        return original, target

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    project_dir = './'
    dataset_path = os.path.join(project_dir, 'datasets', 'mvtec')
    transform = transforms.Compose([
        # transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = MVTecAD(dataset_path, 'wood256', train=True, transform=transform, size_data=(256, 256, 3),
                       multi_mode=True)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
    image, reimage, noisy, mask, _ = next(iter(data_loader))
    # image, reimage, noisy, mask = trainset[0]

    os.makedirs('./sample/', exist_ok=True)
    torchvision.utils.save_image(image[2], './sample/sample_image.png', nrow=3, normalize=True)
    torchvision.utils.save_image(reimage[2], './sample/sample_input.png', nrow=3, normalize=True)
    torchvision.utils.save_image(noisy[2], './sample/sample_noisy.png', nrow=3, normalize=True)
    torchvision.utils.save_image(mask[2], './sample/sample_mask.png', nrow=3, normalize=True)

    # for i, img in enumerate(data_loader):
    #     print(i, type(img[1]))
