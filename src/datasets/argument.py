import os
import math
import numpy as np
from PIL import Image
import cv2
import glob
import copy
import random

import torch
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
import matplotlib.pyplot as plt


class NoiseGrinder():
    
    def __init__(mask_config, size_data, sgm):
        self.mode = mask_config.mode
        if self.mode == "random":
            self.categorys = mask_config.category.split('-')
            self.colors = mask_config.color.split('-')
        elif self.mode == "single":
            self.category = mask_config.category
            self.color = mask_config.color

        self.color = mask_config.color # white, black, mean, mixture
        self.size_data = size_data
        self.mask_size = (mask_config.mask_h, mask_config.mask_w)
        self.ratio = mask_config.mask_ratio
        self.degree = mask_config.line.degree
        self.thickness = mask_config.line.thickness
        self.sgm = sgm


    def __call__(original, input):
        if self.mode == "single":
            generate_noise(self.category, self.color, original, input)
        elif self.mode == "random":
            index_category = random.randint(0, len(self.categorys))
            index_color = random.randint(0, len(self.colors))
            generate_noise(self.categorys[index_category], self.colors[index_color], original, input)


    def grind_noise(category, color, original, input):

        if category == "gauss":
            input, mask = _add_gauss(input)
        elif category == "rec":
            input, mask = _add_rectangle(input, original, color)
        elif category == "cir":
            input, mask = _add_circle(input, color)
        elif category == "line":
            input, mask = _add_line(input, color)


    def _add_gauss(input):
        noise = self.sgm / 255.0 * np.random.randn(self.size_data[0], self.size_data[1], self.size_data[2])
        input = input + noise
        return input, noise


    def _add_rectangle(input, original, color):
        height = self.mask_size[0]
        width = self.mask_size[1]
        size_data = self.size_data
        ratio = self.ratio
        num_sample = int(size_data[0] * size_data[1] * ((1 - ratio)/(width*height+1)))
        loop = 1
        
        mask = np.ones(size_data, np.float32) - 1e-5
        # fill = np.zeros(size_data, np.float32)
        for ch in range(loop):
            idy_mask = np.random.randint(0, size_data[0]-height, num_sample)
            idx_mask = np.random.randint(0, size_data[1]-width, num_sample)

            idx_mask = [idx_mask + i%width for i in range(height*width)]
            idy_mask = [idy_mask + i//width for i in range(height*width)]

            if color == 'c':
                mask[idy_mask, idx_mask, random.randint(0,size_data[2])] = 0 + 1e-5
            elif color == 'b':
                mask[idy_mask, idx_mask, :] = 0 + 1e-5
            elif color == 'm2':
                mask[idy_mask, idx_mask, 0] = np.mean(original[idy_mask, idx_mask, 0])
                mask[idy_mask, idx_mask, 1] = np.mean(original[idy_mask, idx_mask, 1])
                mask[idy_mask, idx_mask, 2] = np.mean(original[idy_mask, idx_mask, 2])
        
        input = input*mask
        return input, mask


    def _add_circle(input, color):
        radius = self.mask_size[0]/2
        overcoat = self.mask_overcoat
        size_data = self.size_data
        ratio = self.ratio
        num_sample = int(size_data[0] * size_data[1] * ((1-ratio)/(math.pi*(radius)**2)))
        loop = 1

        mask = np.ones(size_data, np.float32) - 1e-5
        for ch in range(loop):
            idy_mask = np.random.randint(radius, size_data[0]-radius, num_sample)
            idx_mask = np.random.randint(radius, size_data[1]-radius, num_sample)

            for idy,idx in zip(idy_mask, idx_mask):
                # mask[idy:idy+h, idx:idx+w, ch] = 0
                cv2.circle(mask, (idx, idy), radius, (0,0,0), thickness=-1)
        
        input = input*mask
        return input, mask


    def _add_line(input, color):
        def getXY(r, degree):
            # 度をラジアンに変換
            rad = math.radians(degree)
            x = r * math.cos(rad)
            y = r * math.sin(rad)
            print(x, y)
            return x, y
        length = self.mask_size[0]
        
        thickness = self.thickness
        ratio = self.ratio
        num_sample = int(size_data[0] * size_data[1] * ((1-ratio)/(length * thickness)))
        loop = 1

        mask = np.ones(size_data, np.float32) - 1e-5
        for ch in range(loop)
            idy_mask = np.random.randint(0, size_data[0]-length, num_sample)
            idx_mask = np.random.randint(0, size_data[1]-length, num_sample)

            for idy,idx in zip(idy_mask, idx_mask):
                if "-" not in self.degree
                    degree = self.degree
                else:
                    degree = degree.split("-")
                    degree = random.randint(int(degree[0]), int(degree[1]))
                xy = getXY(length, degree)
                cv2.line(mask, (idx, idy), (xy[0]+idx, xy[1]+idy), (0,0,0))
        input = input*mask
        return input, mask

    def _color_pick(original, mask, color):
        if color == 'm2':
            np.mean(original[mask==0])
        

