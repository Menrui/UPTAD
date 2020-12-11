import os
import sys
import hydra
import random
import numpy as np
import torch

import pytorch_lightning as pl

from lit_SimpleRecon import SimpleReconstructionModule
from lit_MaskRecon import MaskReconstructionModule 

from logging import getLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.datasets.MVTecAD_litmodule import MVTecADDataModule
from src.datasets.YAMAHA_litmodule import YAMAHADataModule
from src.modules import find_module_using_name
from src.data_module import get_datamodule
from src.initializer import init_weight
from src.callbacks.history import HistoryCallback
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
    
    if config.model.mask.add_loss_mask and (config.model.loss_type == 'Texture' or config.model.loss_type == 'Style'):
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
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
    logger.info('cuDNN enabled is {}'.format(torch.backends.cudnn.enabled))
    # }}}

    # Configure train_data loader. {{{
    # =====
    logger.info('Configure training data loader...')
    data_module = get_datamodule(config=config)
    # }}}

    model = find_module_using_name('src.modules.{}'.format(config.model.name), config)
    init_weight(config, model)
    if not config.model.mask.add_mask:
        lit_module = SimpleReconstructionModule(config, model)
        logger.info("SimpleReconstructionModule Setup")
    else:
        lit_module = MaskReconstructionModule(config, model)
        logger.info("MaskReconstructionModule Setup")
    trainer = Trainer(
        gpus=[0],
        deterministic=True,
        # fast_dev_run=True,
        # weights_summary='full',
        max_epochs=config.mode.num_epochs,
        amp_backend='native',
        profiler='simple',
        # progress_bar_refresh_rate=0,
        # log_gpu_memory=True,
        check_val_every_n_epoch=config.mode.validation_epoch,
        callbacks=[ModelCheckpoint(
                        dirpath=str(os.path.join(config.log.training_output_dir, 'checkpoints')),
                        verbose=True,
                        save_last=True,
                        filename='training-{epoch:02d}',
                        period=config.mode.checkpoint_epoch
                    ),
                    HistoryCallback(os.getcwd())
                ]
    )
    trainer.fit(lit_module, data_module)
    trainer.test()


if __name__ == '__main__':
    main_test()
