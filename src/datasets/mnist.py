import os
import numpy as np
import cv2
import copy
from PIL import Image

import torch
from torchvision import datasets


class MNIST(datasets.MNIST):

    def __init__(self, root, normal_digit=None,
                 size_data=(32, 32, 1), **kwargs):
        super(MNIST, self).__init__(root, **kwargs)
        self.normal_digit = normal_digit

        self.sgm = sgm
        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window

        if normal_digit is not None and self.train == True:
            self.split_data()
        elif normal_digit is None and self.train == False:
            pass

    def split_data(self):
        normal_indices = ((self.targets == self.normal_digit).nonzero())
        abnormal_indices = ((self.targets != self.normal_digit).nonzero())

        self.normal_data = self.data[normal_indices]
        self.normal_targets = self.targets[normal_indices]

        self.abnormal_data = self.data[abnormal_indices]
        self.abnormal_targets = self.targets[abnormal_indices]

        self.data = torch.squeeze(self.normal_data)
        self.targets = self.normal_targets

    def __getitem__(self, index):
        original, target = self.data[index].numpy() / 255, int(self.targets[index])
        img = original

        # if original.ndim == 2:
        #     original = np.expand_dims(original, axis=2)
        if original.shape != self.size_data:
            original = cv2.resize(original, (self.size_data[0], self.size_data[1]))

        original = np.expand_dims(original, axis=2)
        label = original + self.noise[index]

        if self.transform:
            original = self.transform((original * 255).astype(np.uint8))
            
        return original, target



def test_dataset():
    import torchvision
    from torchvision import transforms
    _sum = []
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    for i in range(10):
        train_set = MNIST(root='datasets', normal_digit=i,
                          download=True, transform=transform)
        _sum.append(len(train_set))
        print(i, len(train_set), np.unique(train_set.targets), np.sum(_sum))

        loader = torch.utils.data.DataLoader(
            train_set, batch_size=100
        )

        for batch in loader:
            image, target = batch
            print(image.shape, target)
            torchvision.utils.save_image(
                image.to('cpu')[:64], padding=2,
                normalize=True, fp='{}.png'.format(i)
            )
            break

    test_set = MNIST(root='datasets', train=False,
                     download=True, transform=transform)
    print(len(test_set), np.unique(test_set.targets))
    loader = torch.utils.data.DataLoader(
        test_set, batch_size=100)
    for batch in loader:
        image, target = batch
        print(image.shape, target)
        torchvision.utils.save_image(
            image.to('cpu')[:64], padding=2, normalize=True,
            fp='test.png'
        )
        break


if __name__ == '__main__':
    test_dataset()
