# import os
# import numpy as np
# import cv2
# import copy
# from PIL import Image
#
# import torch
# from torchvision import datasets
#
#
# class PyramidGenerator():
#     def __init__(self, batch_size, sgm=25, ratio=0.9, size_data=(256, 256, 3), size_window=(5, 5)):
#         self.sgm = sgm
#         self.ratio = ratio
#         self.size_data = size_data
#         self.size_window = size_window
#
#         self.noise = self.sgm / 255.0 * np.random.randn(
#             len(batch_size), self.size_data[0], self.size_data[1], self.size_data[2])
#
#     def generate_mask(self, inputs: np.ndarray):
#         ratio = self.ratio
#         size_window = self.size_window
#         size_data = inputs.shape[1:] if inputs.ndim == 4 else inputs.shape
#         num_sample = int(size_data[0] * size_data[1] * (1 - ratio))
#
#         masks = np.ones(size_data)
#         outputs = inputs
#
#         for input in inputs:
#             for ich in range(size_data[2]):
#                 # mask is the pixels predicted by N2V
#                 idy_mask = np.random.randint(0, size_data[0], num_sample)
#                 idx_mask = np.random.randint(0, size_data[1], num_sample)
#
#                 # neigh is the pixel that replaces the mask point
#                 idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
#                                               size_window[0] // 2 + size_window[0] % 2,
#                                               num_sample)
#                 idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
#                                               size_window[1] // 2 + size_window[1] % 2,
#                                               num_sample)
#
#                 idy_mask_neigh = idy_mask + idy_neigh
#                 idy_mask_neigh = idy_mask_neigh + (idy_mask_neigh < 0) * size_data[0] - \
#                                  (idy_mask_neigh >= size_data[0]) * size_data[0]
#                 idx_mask_neigh = idx_mask + idx_neigh
#                 idx_mask_neigh = idx_mask_neigh + (idx_mask_neigh < 0) * size_data[1] - \
#                                  (idx_mask_neigh >= size_data[1]) * size_data[1]
#
#                 id_msk = (idy_mask, idx_mask, ich)
#                 id_msk_neigh = (idy_mask_neigh, idx_mask_neigh, ich)
#
#                 outputs[id_msk] = input[id_msk_neigh]
#                 masks[id_msk] = 0.0
#
#         return outputs, masks
