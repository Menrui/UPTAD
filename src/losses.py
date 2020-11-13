from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_ssim
from partialconv.loss import VGG16PartialLoss

def get_loss(config):
    if config.model.loss_type == 'L1':
        return nn.L1Loss()
    elif config.model.loss_type == 'L2':
        return nn.MSELoss()
    elif config.model.loss_type == 'SSIM':
        return pytorch_ssim.SSIM()
    elif config.model.loss_type == 'Style' or config.model.loss_type == 'Perceptual' or config.model.loss_type == 'Texture':
        return VGGLoss(config.model.loss_type)


class VGGLoss(nn.Module):
    def __init__(self, loss_type, l1_alpha=5.0, perceptual_alpha=0.05, style_alpha=120, smooth_alpha=0, feat_num=3):
        super(VGGLoss, self).__init__()
        assert loss_type == 'Style' or loss_type == 'Perceptual' or loss_type == 'Texture'
        self.loss = VGG16PartialLoss(l1_alpha=l1_alpha, perceptual_alpha=perceptual_alpha, style_alpha=style_alpha,
                                     smooth_alpha=smooth_alpha, feat_num=feat_num)
        self.loss_type = loss_type
    
    def __call__(self, output, label):
        tot, vgg_loss, style_loss = self.loss(output, label)
        if self.loss_type == 'Style':
            return style_loss
        elif self.loss_type == 'Perceptual':
            return vgg_loss
        elif self.loss_type == 'Texture':
            return tot



class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class BCELoss(GANLoss):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(BCELoss, self).__init__(target_real_label, target_fake_label)
        self.loss = nn.BCELoss()

    def __ceil__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class LSGANLoss(GANLoss):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(LSGANLoss, self).__init__(target_real_label, target_fake_label)
        self.loss = nn.MSELoss()

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def __call__(self, input, target_is_real, is_generator=False):
        if is_generator:
            return - torch.mean(input)

        if target_is_real:
            return torch.mean(torch.relu(1. - input))
        else:
            return torch.mean(torch.relu(1. + input))


class WGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(WGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if target_is_real:
            loss = -input.mean()
        else:
            loss = input.mean()
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


def compute_gradient_penalty(D, real_data, fake_data, device,
                             type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.Tensor(np.random.random(
                (real_data.size(0), 1, 1, 1))).to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        d_interpolates = D(interpolatesv)
        fake = torch.ones(d_interpolates.size()).to(device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolatesv,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        return gradient_penalty, gradients
    else:
        return 0.0, None
