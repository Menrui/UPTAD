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


class MVTecAD(torch.utils.data.Dataset):

    def __init__(self, root, category, train: bool, maskconf, transform=None, sgm=25, size_data=(256, 256, 3), 
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

        self.sgm = sgm
        self.size_data = size_data
        self.is_loss_mask = maskconf.add_loss_mask
        self.ratio = maskconf.mask_ratio
        self.mask_mode = maskconf.mask_mode
        self.mask_fill = maskconf.mask_fill
        self.mask_size = (maskconf.mask_h, maskconf.mask_w)
        self.mask_overcoat = maskconf.mask_overcoat
        self.add_gauss = maskconf.add_gauss

        self.train_dir = os.path.join(root, category, 'train')
        self.test_dir = os.path.join(root, category, 'test')
        self.gt_dir = os.path.join(root, category, 'ground_truth')

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
                    labels.extend(len(paths) * [1])

            self.img_paths = img_paths
            self.labels = labels
            self.gt_paths = gt_paths

        self.noise = self.sgm / 255.0 * np.random.randn(
            len(self.img_paths), self.size_data[0], self.size_data[1], self.size_data[2])

    def __getitem__(self, index):
        """
        :return:
            original:    original image
            input:  input data to the model
            label:  original image + noise
            mask:   blind spot index
        """
        img_path, target = self.img_paths[index], self.labels[index]
        if not self.train:
            gt_path = self.gt_paths[index]
            ground_truth = plt.imread(gt_path, 0)
            if 'good' in self.gt_paths[index]:
                ground_truth = np.zeros([self.size_data[0], self.size_data[1]])
            ground_truth = np.expand_dims(ground_truth, axis=2)
        else:
            ground_truth = np.zeros([self.size_data[0], self.size_data[1], 1])

        # original = np.array(Image.open(img_path).convert('RGB')) if self.size_data[2]==3 else np.array(Image.open(img_path).convert('L'))
        # original = original/255.0
        original = plt.imread(img_path)

        original = np.expand_dims(original, axis=2) if original.ndim==2 else original
        label = original + self.noise[index]
        # _input = copy.deepcopy(original) if not self.add_gauss else copy.deepcopy(label)
        
        _input  = copy.deepcopy(original)
        if self.mask_mode == 'n2v':
            input, mask = self.n2v_generate_mask(_input)
        else:
            input, mask = self.generate_mask(original, _input, index)
        if not self.is_loss_mask:
            mask = np.zeros(self.size_data, np.float32)

        # Image.fromarray((mask*255).astype(np.uint8).squeeze()).save(f"/home/inagaki/workspace/denoising_ad_mask/sample/{index}.png")
        if self.transform:
            # original = self.transform(Image.fromarray((original*255).astype(np.uint8).squeeze()))
            # input = self.transform(Image.fromarray((input*255).astype(np.uint8).squeeze()))
            # label = self.transform(Image.fromarray((label*255).astype(np.uint8).squeeze()))
            # mask = self.transform(Image.fromarray((mask*255).astype(np.uint8).squeeze()))
            original, input, label, mask, ground_truth = self.transform([original, input, label, mask, ground_truth])

        return original, input, label, mask, ground_truth, target

    def __len__(self):
        return len(self.img_paths)

    def generate_mask(self, original, input, index):
        width = self.mask_size[0]
        height = self.mask_size[1]
        overcoat = self.mask_overcoat
        size_data = self.size_data
        ratio = self.ratio
        num_sample = int(size_data[0] * size_data[1] * ((1 - ratio)/(width*height)))
        # loop = size_data[2] if self.mask_mode=='chole' else 1 if self.mask_mode=='hole' else size_data[2]
        loop = size_data[2]
        
        mask = np.ones(size_data, np.float32)
        fill = np.zeros(size_data, np.float32)
        for ch in range(loop):
            idy_mask = np.random.randint(0, size_data[0]-height, num_sample)
            idx_mask = np.random.randint(0, size_data[1]-width, num_sample)

            idx_mask = [idx_mask + i%width for i in range(height*width)]
            idy_mask = [idy_mask + i//width for i in range(height*width)]

            if self.mask_mode == 'chole':
                mask[idy_mask, idx_mask, ch] = 0
            else:
                mask[idy_mask, idx_mask, :] = 0

            # if not overcoat:
            #     pass
            # for idy,idx in zip(idy_mask, idx_mask):
            #     # w = int(np.random.randint(width[0], width[1])) if type(width)==tuple else width
            #     # h = int(np.random.randint(height[0], height[1])) if type(height)==tuple else height
            #     w = width
            #     h = height
            #     if self.mask_mode=='chole':
            #         mask[idy:idy+h, idx:idx+w, ch] = 0
            #         # fill[idy:idy+h, idx:idx+w, ch] = np.mean(original[idy:idy+h, idx:idx+w, ch])
            #     else:
            #         mask[idy:idy+h, idx:idx+w, :] = 0
            #         # if self.mask_fill == 'lmean':
            #         #     fill[idy:idy+h, idx:idx+w, 0] = np.mean(original[idy:idy+h, idx:idx+w, 0])
            #         #     fill[idy:idy+h, idx:idx+w, 1] = np.mean(original[idy:idy+h, idx:idx+w, 1])
            #         #     fill[idy:idy+h, idx:idx+w, 2] = np.mean(original[idy:idy+h, idx:idx+w, 2])
            #         # elif self.mask_fill == 'chmean':
            #         #     fill[idy:idy+h, idx:idx+w, :] = np.mean(original[idy:idy+h, idx:idx+w, :])
            #         # else:
            #         #     pass

        # mask, fill = calc_mask(loop=loop, num_sample=num_sample, size_data=size_data, width=width, height=height, 
        #                             mask_mode=self.mask_mode, mask_fill=self.mask_fill)
        if self.mask_mode=='normal':
            input = input + self.noise[index]*(1-mask)
        else:
            input = input*mask
            # if self.mask_fill is not None and 'mean' in self.mask_fill:
            #     input = input + fill
        return input, mask

    def n2v_generate_mask(self, input):
        ratio = self.ratio
        # size_window = self.size_window
        # size_data = self.size_data
        size_window = (5,5)
        size_data = input.shape
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[2]):
            # mask is the pixels predicted by N2V
            idy_mask = np.random.randint(0, size_data[0], num_sample)
            idx_mask = np.random.randint(0, size_data[1], num_sample)

            # neigh is the pixel that replaces the mask point
            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
                                          size_window[0] // 2 + size_window[0] % 2,
                                          num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
                                          size_window[1] // 2 + size_window[1] % 2,
                                          num_sample)

            idy_mask_neigh = idy_mask + idy_neigh
            idy_mask_neigh = idy_mask_neigh + (idy_mask_neigh < 0) * size_data[0] - \
                             (idy_mask_neigh >= size_data[0]) * size_data[0]
            idx_mask_neigh = idx_mask + idx_neigh
            idx_mask_neigh = idx_mask_neigh + (idx_mask_neigh < 0) * size_data[1] - \
                             (idx_mask_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_mask, idx_mask, ich)
            id_msk_neigh = (idy_mask_neigh, idx_mask_neigh, ich)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask


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
