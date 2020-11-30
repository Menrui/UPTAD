import os
import math
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


def generate_noise(mask_config, size_data, original, input):
    mode = mask_config.mode
    category = mask_config.category
    color = mask_config.color # white, black, mean, mixture

    if mode == "assign":
        if category == "Gaussian":
            pass
        elif category == "Rectangle":
            pass
        elif category == "Circle":
            pass
        elif category == "Line":
            pass
        elif category == "Structual":
            pass
        elif category == "All"
    elif mode == "random":
        categorys = 0
        https://note.nkmk.me/python-random-choice-sample-choices/
        pass


def add_gauss(input, size_data, sigma):
    noise = sigma / 255.0 * np.random.randn(size_data[0], size_data[1], size_data[2])
    input = input + noise
    return input


def add_rectangle(input, color, size_data, mask_size, ratio):
    width = mask_size[0]
    height = mask_size[1]
    size_data = size_data
    ratio = ratio
    num_sample = int(size_data[0] * size_data[1] * ((1 - ratio)/(width*height+1)))
    loop = size_data[2]
    
    mask = np.ones(size_data, np.float32) - 1e-5
    # fill = np.zeros(size_data, np.float32)
    for ch in range(loop):
        idy_mask = np.random.randint(0, size_data[0]-height, num_sample)
        idx_mask = np.random.randint(0, size_data[1]-width, num_sample)

        idx_mask = [idx_mask + i%width for i in range(height*width)]
        idy_mask = [idy_mask + i//width for i in range(height*width)]

        if self.mask_mode == 'chole':
            mask[idy_mask, idx_mask, ch] = 0 + 1e-5
        else:
            mask[idy_mask, idx_mask, :] = 0 + 1e-5
    
    input = input*mask
    
    return input, mask


def add_circle(input, color, size_data, mask_size, ratio):
    width = mask_size[0]
    height = mask_size[1]
    size_data = size_data
    ratio = ratio
    num_sample = int(size_data[0] * size_data[1] * ((1-ratio)/(math.pi*(width/2)**2)))
    loop = size_data[2]

    mask = np.ones(size_data, np.float32) - 1e-5
    for ch in range(loop):
        idy_mask = np.random.randint(0, size_data[0]-height, num_sample)
        idx_mask = np.random.randint(0, size_data[1]-width, num_sample)

        for idy,idx in zip(idy_mask, idx_mask):
            # mask[idy:idy+h, idx:idx+w, ch] = 0
            cv2.circle(mask, (idx, idy), width/2, (0,0,0), thickness=-1)
