import os
import sys
import time
import numpy as np
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch

# from src.config import save_config
from logging import getLogger

# output_dir = os.getcwd()
logger = getLogger('root')


# def switch_debug_mode(config):
#     return config

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_logdirs(config, output_dir):
    # Training. {{{
    # =====
    config.log.training_output_dir = os.path.join(output_dir, config.log.training_output_dir)
    config.log.trained_model_dir = os.path.join(
        config.log.training_output_dir, config.log.trained_model_dir)
    config.log.vis_dir = os.path.join(output_dir, config.log.vis_dir)

    makedirs(config.log.training_output_dir)
    makedirs(config.log.trained_model_dir)
    makedirs(config.log.vis_dir)
    # }}}

    # Test. {{{
    # =====
    config.log.test_output_dir = os.path.join(output_dir, 'test')
    makedirs(config.log.test_output_dir)
    # }}}

    config.log.output_dir = output_dir
    if config.mode.use_tb:
        config.log.summary_path = os.path.join(output_dir)

    # save_config(config, config.log.output_dir, filename='train_config.yaml')
    # save_config(config, config.log.output_dir, filename='test_config.yaml')

    return config


class AverageMeter(object):
    def __init__(self):
        self.reset()

    @property
    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n


class History(object):

    def __init__(self, keys, output_dir):
        self.output_dir = output_dir
        self.keys = keys

        self.logs = {key: [] for key in keys}

    def __call__(self, data):
        for key, value in data.items():
            self.logs[key].append(value)

    def save(self, filename='history.pkl'):
        savepath = os.path.join(self.output_dir, filename)
        with open(savepath, 'wb') as f:
            pickle.dump(self.logs, f)

    def plot_graph(self, filename='loss.png'):
        fig, ax = plt.subplots()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training curve.')

        for i, (key, value) in enumerate(self.logs.items()):
            x = np.arange(len(value))
            y = np.array(value)
            ax.plot(x, y, label=key, color=cm.cividis(i / len(self.logs.keys())))

        ax.legend(loc='best')

        save_path = os.path.join(self.output_dir, filename)
        logger.info('Save {}'.format(save_path))
        plt.savefig(save_path, transparent=True)
        plt.clf()
        plt.cla()
        plt.close('all')


def save_checkpoint(config, model, optimizer, filename):
    logger.info('Save {}'.format(filename))
    save_dir = config.log.trained_model_dir

    save_path = os.path.join(save_dir, filename)
    torch.save({
        'model': model.state_dict(),
        'opt': optimizer.state_dict()
    }, save_path)


def save_model(config, model, filename):
    logger.info('Save {}'.format(filename))
    save_dir = config.log.trained_model_dir

    save_path = os.path.join(save_dir, filename)
    torch.save({
        'model': model.state_dict(),
    }, save_path)


def load_checkpoint(config, model, optimizer, filename):
    logger.info('Load {} model trained after {} epochs'.format(
        model.__class__.__name__, config.mode.checkpoint_epochs))

    # print(os.path.join(config.log.trained_output_dir, config.log.trained_model_dir, filename))
    load_path = os.path.join(config.log.output_dir, config.log.training_output_dir, config.log.trained_model_dir, filename)
    ckpt = torch.load(load_path, map_location=lambda storage, pos: storage)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['opt'])
    return model, optimizer


def load_model(config, model, filename):
    logger.info('Load {} model trained after {} epochs'.format(
        model.__class__.__name__, config.mode.checkpoint_epochs))

    load_path = os.path.join(config.log.output_dir, config.log.training_output_dir, config.log.trained_model_dir, filename)
    ckpt = torch.load(load_path, map_location=lambda storage, pos: storage)
    model.load_state_dict(ckpt['model'])
    return model


def clear_fig():
    plt.clf()
    plt.cla()
    plt.close('all')


from mpl_toolkits import axes_grid1


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def make_patches(x_tensor, config):
    x_tensor = x_tensor.unfold(2, config.dataset.patch_size, config.dataset.patch_step).unfold(3,
                                                                                               config.dataset.patch_size,
                                                                                               config.dataset.patch_step)
    row, col = x_tensor.size(3), x_tensor.size(2)
    x_tensor = x_tensor.permute([0, 2, 3, 1, 4, 5]).reshape(-1, 3, config.dataset.patch_size, config.dataset.patch_size)
    return x_tensor, row, col


def labeling_patchs(ground_truth_patches_tensor):
    assert ground_truth_patches_tensor.ndim == 4

    def labeling(patches):
        labels = []
        for patch in patches:
            if np.all(patch == 0):
                labels.extend([0])
            else:
                labels.extend([1])
        return labels

    ground_truth_patches = ground_truth_patches_tensor.detach().cpu().numpy().transpose(0, 3, 1, 2)
    labels = labeling(ground_truth_patches)

    return torch.tensor(labels)
