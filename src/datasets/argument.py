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
    
    def __init__(self, mask_config, size_data, noise_sgm):
        self.mode = mask_config.mode
        if self.mode == "random":
            self.categorys = mask_config.category.split('-')
            self.colors = mask_config.color.split('-')
        elif self.mode == "single":
            self.category = mask_config.category
            self.color = mask_config.color

        # self.color = mask_config.color # white, black, mean, mixture
        self.size_data = size_data
        self.mu = mask_config.mu
        self.sigma = mask_config.sigma

        # self.mask_size = (mask_config.mask_h, mask_config.mask_w)
        self.mask_size = mask_config.size
        self.ratio = mask_config.mask_ratio
        self.degree = mask_config.line.degree
        self.thickness = mask_config.line.thickness
        self.noise_sgm = noise_sgm


    def __call__(self, original, input):
        if self.mode == "single":
            input, mask = self.grind_noise(self.category, self.color, original, input)
        elif self.mode == "random":
            index_category = random.randint(0, len(self.categorys)-1)
            index_color = random.randint(0, len(self.colors)-1)
            input, mask = self.grind_noise(self.categorys[index_category], self.colors[index_color], original, input)
        else:
            assert False, "Invalid mode :{}".format(self.mode)

        return input, mask


    def grind_noise(self, category, color, original, input):

        if category == "gauss":
            input, mask = self._add_gauss(input)
        elif category == "rec":
            input, mask = self._add_rectangle(input, original, color)
        elif category == "cir":
            input, mask = self._add_circle(input, original, color)
        elif category == "line":
            input, mask = self._add_line(input, original, color)
        elif category == "stain":
            input, mask = self._add_l_ellipse(input, original)
        else:
            assert False, "Invalid category {}".format(category)
        
        return input, mask


    def _add_gauss(self, input):
        noise = self.sgm / 255.0 * np.random.randn(self.size_data[0], self.size_data[1], self.size_data[2])
        input = input + noise
        return input, noise


    def _add_rectangle(self, input, original, color):
        # height = self.mask_size[0]
        # width = self.mask_size[1]
        mask_size = self.mask_size
        size_data = self.size_data
        ratio = self.ratio
        # num_sample = int(size_data[0] * size_data[1] * ((1 - ratio)/(width*height+1)))
        num_sample = int(size_data[0] * size_data[1] * ((1 - ratio)/(mask_size**2+1)))
        loop = 1
        
        mask = np.ones(size_data, np.float32) - 1e-5
        fill = np.zeros(size_data, np.float32)
        for ch in range(loop):
            idy_mask = np.random.randint(0, size_data[0]-mask_size, num_sample)
            idx_mask = np.random.randint(0, size_data[1]-mask_size, num_sample)

            for idy,idx in zip(idy_mask, idx_mask):
                _random = round(random.normalvariate(mu=self.mu, sigma=self.sigma))
                w = abs(mask_size + _random)
                h = abs(mask_size + _random)
                if color=='c':
                    mask[idy:idy+h, idx:idx+w, random.randint(0,size_data[2]-1)] = 0
                elif color == 'b':
                    mask[idy:idy+h, idx:idx+w, :] = 0
                elif color == 'm2':
                    mask[idy:idy+h, idx:idx+w, :] = 0
                    fill[idy:idy+h, idx:idx+w, 0] = np.mean(original[idy:idy+h, idx:idx+w, 0])
                    fill[idy:idy+h, idx:idx+w, 1] = np.mean(original[idy:idy+h, idx:idx+w, 1])
                    fill[idy:idy+h, idx:idx+w, 2] = np.mean(original[idy:idy+h, idx:idx+w, 2])
                elif color == 'm1':
                    mask[idy:idy+h, idx:idx+w, :] = 0
                    fill[idy:idy+h, idx:idx+w, :] = np.mean(original[idy:idy+h, idx:idx+w, :])
        
        input = input*mask + fill
        return input, mask


    def _add_circle(self, input, original, color):
        radius = round(self.mask_size/2)
        size_data = self.size_data
        ratio = self.ratio
        num_sample = int(size_data[0] * size_data[1] * ((1-ratio)/(math.pi*(radius)**2)))
        loop = 1

        mask = np.zeros(size_data, np.float32) - 1e-5
        for ch in range(loop):
            idy_mask = np.random.randint(radius, size_data[0]-radius, num_sample)
            idx_mask = np.random.randint(radius, size_data[1]-radius, num_sample)

            for idy,idx in zip(idy_mask, idx_mask):
                # mask[idy:idy+h, idx:idx+w, ch] = 0
                tmp_mask = np.zeros(size_data, np.float32)
                _radius = radius + round(random.normalvariate(mu=self.mu, sigma=self.sigma)/2)
                _radius = abs(_radius)
                cv2.circle(tmp_mask, (idx, idy), _radius, color=self._cv_color_pick(color, original), thickness=-1)
                mask = (mask + tmp_mask)
        
        mask = 1-np.where(mask>=0.5,1,0)
        input = input*mask
        return input, mask


    def _add_line(self, input, original, color):
        def getXY(r, degree):
            # 度をラジアンに変換
            rad = math.radians(degree)
            x = r * math.cos(rad)
            y = r * math.sin(rad)
            # print(x, y)
            return round(x), round(y)
        length = self.mask_size
        thickness = self.thickness
        size_data = self.size_data
        ratio = self.ratio
        num_sample = int(size_data[0] * size_data[1] * ((1-ratio)/(length * thickness)))
        loop = 1

        mask = np.zeros(size_data, np.float32) - 1e-5
        for ch in range(loop):
            idy_mask = np.random.randint(0, size_data[0]-length, num_sample)
            idx_mask = np.random.randint(0, size_data[1]-length, num_sample)

            for idy,idx in zip(idy_mask, idx_mask):
                if type(self.degree) is int:
                    degree = self.degree
                else:
                    degree = self.degree.split("-")
                    degree = random.randint(int(degree[0]), int(degree[1]))
                _length = abs(length + round(random.normalvariate(mu=self.mu, sigma=self.sigma)))
                xy = getXY(_length, degree)
                
                tmp_mask = np.zeros(size_data, np.float32)
                cv2.line(tmp_mask, (idx, idy), (xy[0]+idx, xy[1]+idy), color=self._cv_color_pick(color, original), thickness=thickness)
                mask = (mask + tmp_mask)

        mask = 1-np.where(mask>=0.5,1,0)
        input = input*mask
        return input, mask

    def _add_l_ellipse(self, input, original):
        def adjust(img, alpha=1.0, beta=0.0):
            dst = alpha * img + beta/255
            return np.clip(dst, 0., 1.)

        def getXY(r, degree):
            # 度をラジアンに変換
            rad = math.radians(degree)
            x = r * math.cos(rad)
            y = r * math.sin(rad)
            # print(x, y)
            return round(x), round(y)
        
        mask_size = self.mask_size
        ratio = self.ratio
        size_data = self.size_data
        num_sample = random.randint(1,3)
        area = int((size_data[0] * size_data[1] * (1-ratio))/num_sample)

        mask = np.ones(size_data, np.float32) - 1e-5
        fill = np.copy(original)
        fill = adjust(fill, alpha=random.random()*2, beta=random.randint(-10, 10))
        fill = cv2.GaussianBlur(fill, (25,25), 0)

        for i in range(num_sample):
            short_side = self.mask_size*10 + round(random.normalvariate(mu=self.mu, sigma=self.sigma))*10
            long_side = int(area/short_side * math.pi)
            degree = random.randint(0,180)
            long_xy = getXY(r=long_side, degree=degree)
            short_xy = getXY(r=short_side, degree=degree)
            y = abs(long_xy[1] if long_xy[1]>short_xy[1] else short_xy[1])
            x = abs(long_xy[0] if long_xy[0]>short_xy[0] else short_xy[0])

            idy = random.randint(y, size_data[0]-y) if y < size_data[0]-y else random.randint(size_data[0]-y, y)
            idx = random.randint(x, size_data[1]-x) if x < size_data[1]-x else random.randint(size_data[1]-x, x)

            cv2.ellipse(mask, ((idx,idy), (long_side, short_side), degree), (0,0,0), thickness=-1)

        input = input*mask + fill*(1-mask)
        return input, mask

    def _cv_color_pick(self, color, original):
        if color == 'c':
            select=random.randint(0,2)
            if select == 0:
                pick = (1.,0.,0.)
            elif select == 1:
                pick = (0.,1.,0.)
            elif select == 2:
                pick = (0.,0.,1.)
        elif color == 'b':
            pick = (1,1,1)
 
        return pick

