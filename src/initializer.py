from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import getLogger
from torch.nn import init
import torch

logger = getLogger('root')


def init_weight(config, net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if config.model.init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif config.model.init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif config.model.init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif config.model.init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % config.model.init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    logger.info('Initialize {} with {}'.format(
        net.__class__.__name__, config.model.init_type))
    net.apply(init_func)


# Old initializer
# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1 and classname != 'Conv':
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif classname.find("Linear") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.01)
#         if m.bias is not None:
#             m.bias.data.fill_(0)