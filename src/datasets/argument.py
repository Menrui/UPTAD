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


def generate_noise(mask_config, original, input):
    mode = mask_config.mode
    category = mask_config.category
    color = mask_config.color # white, black, mean, mixture

    if mode == "assign":
        if category == "Gaussian":
            pass
        elif category == "Rectangle":
            pass
        elif category == "Ellipse":
            pass
        elif category == "Line":
            pass
        elif category == "Structual":
            pass
        elif category == "All"
    elif mode == "random":
        https://note.nkmk.me/python-random-choice-sample-choices/
        pass


    
