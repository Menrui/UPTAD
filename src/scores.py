import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

import pytorch_ssim
from partialconv.loss import VGG16PartialLoss

def get_scores(config):
    if config.mode.score_type == 'L1max':
        return L1_score(True)
    elif config.mode.score_type == 'L1mean':
        return L1_score(False)
    elif config.mode.score_type == 'L2max':
        return L2_score(True)
    elif config.mode.score_type == 'L2mean':
        return L2_score(False)
    elif config.mode.score_type == 'SSIM':
        return SSIM_score(window_size=11, is_patch_max_mode=False)
    elif config.mode.score_type == 'PatchSSIM':
        return SSIM_score(window_size=11, is_patch_max_mode=True)
    elif config.mode.score_type == 'Style':
        return VGG_score(config.mode.score_type, is_patch_max_mode=False)
    elif config.mode.score_type == 'Perceptual':
        return VGG_score(config.mode.score_type, is_patch_max_mode=False)
    elif config.mode.score_type == 'SP':
        return VGG_score(config.mode.score_type, is_patch_max_mode=False)
    elif config.mode.score_type == 'L1SP':
        return VGG_score(config.mode.score_type, is_patch_max_mode=False)
    elif config.mode.score_type == 'PatchStyle':
        return VGG_score(config.mode.score_type, is_patch_max_mode=True)
    elif config.mode.score_type == 'PatchPerceptual':
        return VGG_score(config.mode.score_type, is_patch_max_mode=True)
    elif config.mode.score_type == 'PatchSP':
        return VGG_score(config.mode.score_type, is_patch_max_mode=True)
    elif config.mode.score_type == 'PatchL1SP':
        return VGG_score(config.mode.score_type, is_patch_max_mode=True)
    else:
        assert False, 'invalid score-type'

class L1_score(nn.Module):
    def __init__(self, use_max_func):
        super(L1_score, self).__init__()
        self.use_max_func = use_max_func
        
    def __call__(self, input, output):
        res_loss = torch.abs(input - output)
        res_loss = res_loss.view(res_loss.size()[0], -1)
        if self.use_max_func:
            res_loss = torch.max(res_loss, 1)[0]
        else:
            res_loss = torch.mean(res_loss, 1)        
        return res_loss


class L2_score(nn.Module):
    def __init__(self, use_max_func):
        super(L2_score, self).__init__()
        self.use_max_func = use_max_func
        
    def __call__(self, input, output):
        res_loss = (input - output)**2
        res_loss = res_loss.view(res_loss.size()[0], -1)
        if self.use_max_func:
            res_loss = torch.max(res_loss, 1)[0]
        else:
            res_loss = torch.mean(res_loss, 1)        
        return res_loss


class SSIM_score(nn.Module):
    def __init__(self, window_size, is_patch_max_mode):
        super(SSIM_score, self).__init__()
        self.SSIM = pytorch_ssim.SSIM(window_size=window_size)
        self.is_patch_max_mode = is_patch_max_mode

    def __call__(self, input, output):
        if self.is_patch_max_mode:
            input = input.unfold(2,16,16).unfold(3,16,16).permute([0, 2, 3, 1, 4, 5]).reshape(-1, 3, 16, 16)
            output = output.unfold(2,16,16).unfold(3,16,16).permute([0, 2, 3, 1, 4, 5]).reshape(-1, 3, 16, 16)
            ssim_value = self.SSIM(input, output)
            ssim_value = torch.max(ssim_value)
        else:
            ssim_value = self.SSIM(input, output)
        score = torch.unsqueeze(-ssim_value, dim=-1)
        return score


class VGG_score(nn.Module):
    def __init__(self, score_type, is_patch_max_mode, l1_alpha=1, perceptual_alpha=1, style_alpha=1):
        super(VGG_score, self).__init__()
        self.VGG16PartialLoss = VGG16PartialLoss(l1_alpha=l1_alpha, perceptual_alpha=perceptual_alpha, style_alpha=style_alpha)
        self.score_type = score_type
        self.is_patch_max_mode = is_patch_max_mode
    
    def __call__(self, input, output):
        if self.is_patch_max_mode:
            input = input.unfold(2,32,16).unfold(3,32,16).permute([0, 2, 3, 1, 4, 5]).reshape(-1, 3, 32, 32)
            output = output.unfold(2,32,16).unfold(3,32,16).permute([0, 2, 3, 1, 4, 5]).reshape(-1, 3, 32, 32)
            tot, perceptual, style = self.VGG16PartialLoss(output, input)
            tot = torch.max(tot)
            perceptual = torch.max(perceptual)
            style = torch.max(style)
        else:
            tot, perceptual, style = self.VGG16PartialLoss(output, input)
        if self.score_type == 'Style':
            return torch.unsqueeze(style, dim=-1)
        elif self.score_type == 'Perceptual':
            return torch.unsqueeze(perceptual, dim=-1)
        elif self.score_type == 'SP':
            return torch.unsqueeze(style + perceptual, dim=-1)
        elif self.score_type == 'L1SP':
            return torch.unsqueeze(tot, dim=-1)


# if __name__ == '__main__':
#     a = torch.randn(5, 4,4, 3)
#     b = torch.randn(5,4,4,3)
#     L2_score = L2_score(False)

#     norm1 = ((a-b)**2).view((a.size()[0], -1))
#     norm1 = torch.mean(norm1, dim=1)
#     norm2 = L2_score(a, b)

#     print(norm1, norm1.size(), norm2, norm2.size())