import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os
import copy

import torchvision.transforms as transforms
import torchvision

from logging import getLogger

logger = getLogger('root')


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data

        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        if input.ndim == 3:
            original = original.transpose((2, 0, 1)).astype(np.float32)
            input = input.transpose((2, 0, 1)).astype(np.float32)
            label = label.transpose((2, 0, 1)).astype(np.float32)
            mask = mask.transpose((2, 0, 1)).astype(np.float32)
            gt = gt.transpose((2, 0, 1)).astype(np.float32)
        else:
            original = original.transpose((0, 3, 1, 2)).astype(np.float32)
            input = input.transpose((0, 3, 1, 2)).astype(np.float32)
            label = label.transpose((0, 3, 1, 2)).astype(np.float32)
            mask = mask.transpose((0, 3, 1, 2)).astype(np.float32)
            gt = gt.transpose((0, 3, 1, 2)).astype(np.float32)

        return torch.from_numpy(original), torch.from_numpy(input), torch.from_numpy(label), torch.from_numpy(
            mask), torch.from_numpy(gt)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean[0]
        self.std = std[0]

    def __call__(self, data):
        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        original = (original - self.mean) / self.std
        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std

        return original, input, label, mask, gt


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        if np.random.rand() > 0.5:
            original = np.fliplr(original)
            input = np.fliplr(input)
            label = np.fliplr(label)
            mask = np.fliplr(mask)
            gt = np.fliplr(gt)

        if np.random.rand() > 0.5:
            original = np.flipud(original)
            input = np.flipud(input)
            label = np.flipud(label)
            gt = np.flipud(gt)

        return original, input, label, mask, gt


class Resize(object):
    """Rescale the image in a sample to a given size

    Args:
      output_size (tuple or int): Desired output size.
                                  If tuple, output is matched to output_size.
                                  If int, smaller of image edges is matched
                                  to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        h, w = input.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        original = transform.resize(original, (new_h, new_w))
        input = transform.resize(input, (new_h, new_w))
        label = transform.resize(label, (new_h, new_w))
        mask = transform.resize(mask, (new_h, new_w))
        gt = transform.resize(gt, (new_h, new_w))

        return original, input, label, mask, gt


class RandomCrop(object):
    """Crop randomly the image in a sample

    Args:
      output_size (tuple or int): Desired output size.
                                  If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        h, w = input.shape[:2]
        new_h, new_w = self.output_size
        # print(h, w, new_h, new_w)

        if not h == new_h and w == new_w:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
        else:
            top = 0
            left = 0

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        # input = input[top: top + new_h, left: left + new_w]
        # label = label[top: top + new_h, left: left + new_w]

        original = original[id_y, id_x]
        input = input[id_y, id_x]
        label = label[id_y, id_x]
        mask = mask[id_y, id_x]
        gt = gt[id_y, id_x]

        return original, input, label, mask, gt


class UnifromSample(object):
    """Crop randomly the image in a sample

    Args:
      output_size (tuple or int): Desired output size.
                                  If int, square crop is made.
    """

    def __init__(self, stride):
        assert isinstance(stride, (int, tuple))
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            assert len(stride) == 2
            self.stride = stride

    def __call__(self, data):
        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        h, w, c = input.shape
        stride_h, stride_w = self.stride
        assert h % stride_h == 0 and w % stride_w == 0
        # new_h = h//stride_h
        # new_w = w//stride_w

        # top = np.random.randint(0, stride_h + (h - new_h * stride_h))
        # left = np.random.randint(0, stride_w + (w - new_w * stride_w))

        # id_h = np.arange(top, h, stride_h)[:, np.newaxis]
        # id_w = np.arange(left, w, stride_w)

        # original = original[id_h, id_w]
        # input = input[id_h, id_w]
        # label = label[id_h, id_w]
        # mask = mask[id_h, id_w]
        # gt = gt[id_h, id_w]

        # original = np.concatenate([np.hsplit(h_img, h // stride_h) for h_img in np.vsplit(original, w // stride_w)], axis=0)
        # input = np.concatenate([np.hsplit(h_img, h // stride_h) for h_img in np.vsplit(input, w // stride_w)], axis=0)
        # label = np.concatenate([np.hsplit(h_img, h // stride_h) for h_img in np.vsplit(label, w // stride_w)], axis=0)
        # mask = np.concatenate([np.hsplit(h_img, h // stride_h) for h_img in np.vsplit(mask, w // stride_w)], axis=0)
        # gt = np.concatenate([np.hsplit(h_img, h // stride_h) for h_img in np.vsplit(gt, w // stride_w)], axis=0)

        original = np.concatenate([np.hsplit(h_img, w // stride_w) for h_img in np.vsplit(original, h // stride_h)], axis=0)
        input = np.concatenate([np.hsplit(h_img, w // stride_w) for h_img in np.vsplit(input, h // stride_h)], axis=0)
        label = np.concatenate([np.hsplit(h_img, w // stride_w) for h_img in np.vsplit(label, h // stride_h)], axis=0)
        mask = np.concatenate([np.hsplit(h_img, w // stride_w) for h_img in np.vsplit(mask, h // stride_h)], axis=0)
        gt = np.concatenate([np.hsplit(h_img, w // stride_w) for h_img in np.vsplit(gt, h // stride_h)], axis=0)

        # original = np.concatenate([np.array_split(h_img, w // stride_w, 1) for h_img in np.array_split(original, h // stride_h, 0)], axis=0)
        # input = np.concatenate([np.array_split(h_img, w // stride_w, 1) for h_img in np.array_split(input, h // stride_h, 0)], axis=0)
        # label = np.concatenate([np.array_split(h_img, w // stride_w, 1) for h_img in np.array_split(label, h // stride_h, 0)], axis=0)
        # mask = np.concatenate([np.array_split(h_img, w // stride_w, 1) for h_img in np.array_split(mask, h // stride_h, 0)], axis=0)
        # gt = np.concatenate([np.array_split(h_img, w // stride_w, 1) for h_img in np.array_split(gt, h // stride_h, 0)], axis=0)


        return original, input, label, mask, gt


class ZeroPad(object):
    """Rescale the image in a sample to a given size

    Args:
      output_size (tuple or int): Desired output size.
                                  If tuple, output is matched to output_size.
                                  If int, smaller of image edges is matched
                                  to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        h, w = input.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        l = (new_w - w) // 2
        r = (new_w - w) - l

        u = (new_h - h) // 2
        b = (new_h - h) - u

        original = np.pad(original, pad_width=((u, b), (l, r), (0, 0)))
        input = np.pad(input, pad_width=((u, b), (l, r), (0, 0)))
        label = np.pad(label, pad_width=((u, b), (l, r), (0, 0)))
        mask = np.pad(mask, pad_width=((u, b), (l, r), (0, 0)))
        gt = np.pad(gt, pad_width=((u, b), (l, r), (0, 0)))

        return original, input, label, mask, gt


class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        # return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        original = original.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        input = input.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        label = label.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        mask = mask.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        gt = gt.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        return original, input, label, mask, gt


class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # data = self.std * data + self.mean
        # return data
        original, input, label, mask, gt = data[0], data[1], data[2], data[3], data[4]

        original = original * self.std + self.mean
        input = input * self.std + self.mean
        label = label * self.std + self.mean

        return original, input, label, mask, gt
