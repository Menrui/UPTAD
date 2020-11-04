import os
import sys
import hydra
import random
import numpy as np
import torch

import pytorch_lightning as pl

from AutoEncoder import AutoEncoder
from logging import getLogger
from pytorch_lightning import Trainer
from src.loader import get_loader
from src.utils import make_logdirs


@hydra.main(config_path='config', config_name='config')
def main_test(config):
    # Set random seed. {{{
    # ====
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    # }}}

    # config update.{{{
    # =====
    config.work_dir = hydra.utils.get_original_cwd()
    config.workname = os.path.basename(os.getcwd())
    # }}}

    # Make dirnames. {{{
    # =====
    config = make_logdirs(config, output_dir=os.getcwd())
    # }}}

    # Get logger. {{{
    # =====
    logger = getLogger('root')
    logger.info('Start {} training...'.format(config.workname))
    logger.info('Device: {}'.format(config.device))
    # }}}

    # Configure train_data loader. {{{
    # =====
    logger.info('Configure training data loader...')
    train_loader, val_loader = get_loader(config, is_train=True)
    # }}}
    print('t')

    model = AutoEncoder(config)
    trainer = Trainer(
        gpus=[0]
    )
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main_test()
