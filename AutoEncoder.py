import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from pytorch_lightning.core.lightning import LightningModule

from logging import getLogger
from sklearn.metrics import roc_curve, roc_auc_score, auc
from src.modules.ae import AutoEncoder as AE
from src.losses import get_loss
from src.utils import add_colorbar, History

logger = getLogger('root')


class AutoEncoder(LightningModule):
    def __init__(self, config, *args, **kwargs):
        super(self, AutoEncoder).__init__()

        # hyper parameters
        self.learning_rate = config.model.lr
        self.beta_1 = config.model.beta_1
        self.beta_2 = config.model.beta_2

        # network define
        self.model = AE(config)

        # error function define
        self.critation = get_loss(config)

        # test parameter
        self.digit = config.dataset.digit

        # output dir path
        self.train_output_dir = config.log.train_output_dir
        self.vis_output_dir = config.log.vis_output_dir
        self.test_output_dir = config.log.test_output_dir

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        original, input, label, mask, _, _ = batch
        output = self.model(input.float())

        # calculate Loss
        loss_G = self.critation(output * (1 - mask), original * (1 - mask))

        self.log('train_loss', loss_G, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_history({'train_loss': loss_G.item()})
        return loss_G

    def validation_step(self, batch, batch_idx):
        original, input, label, mask, _, _ = batch
        output = self.model(input.float())

        # calculate Loss
        loss_G = self.critation(output * (1 - mask), original * (1 - mask))

        self.log('val_loss', loss_G, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_G

    def test_step(self, batch, batch_idx):
        original, input, _, mask, gt, target = batch

        _size = input.size()
        if input.ndim != 4:
            original = original.reshape(-1, _size[2], _size[3], _size[4])
            input = input.reshape(-1, _size[2], _size[3], _size[4])
            mask = mask.reshape(-1, _size[2], _size[3], _size[4])
            gt = gt.reshape(-1, 1, _size[3], _size[4])  # ground truth mask is grayscale

        output = self.model(original)

        anomaly_score = self._compute_anomaly_score(original, output)

        self.log_dict({'anomaly_score': anomaly_score, 'anomaly_label': target, 'id': batch_idx})
        return anomaly_score, target

    def test_epoch_end(self, outputs):
        """
        :param outputs: anomaly_score, target
        :return:
        """
        # Compute roc curve and auc score. {{{
        # =====
        targets = torch.cat([target for target in outputs[0]]).numpy()
        anomaly_scores = torch.cat([score for score in outputs[1]]).numpy()
        anomaly_labels = np.uint8(targets != self.digit)

        fpr, tpr, th = roc_curve(anomaly_labels, anomaly_scores)
        auc = roc_auc_score(anomaly_labels, anomaly_scores)

        df = pd.DataFrame({'fpr': fpr,
                           'tpr': tpr,
                           'th': th})
        df.to_csv(os.path.join(os.getcwd(), 'training.roc.csv'), index=None)
        logger.info('fpr@tpr=1.0: {}  AUC: {}'.format(fpr[-2], auc))
        # }}}

        # Plot histogram of anomaly scores. {{{
        # ======
        fig, ax = plt.subplots()
        ax.set_xlabel('Anomaly score')
        ax.set_ylabel('Number of Instances')
        ax.set_title('Histogram of anomaly scores')
        normal_scores = anomaly_scores[anomaly_labels == 0]
        abnormal_scores = anomaly_scores[anomaly_labels == 1]
        ax.hist([normal_scores, abnormal_scores], 50, label=['Normal samples', 'Abnormal samples'],
                alpha=0.5, histtype='stepfilled')
        ax.legend()
        ax.grid(which='major', axis='y', color='grey',
                alpha=0.8, linestyle="--", linewidth=1)
        plt.savefig(os.path.join(os.getcwd(), 'training-hist.png'), transparent=True)
        plt.clf()
        # }}}

        # Plot ROC curve. {{{
        # =====
        fig, ax = plt.subplots()
        ax.set_xlabel('FPR: False positive rate')
        ax.set_ylabel('TPR: True positive rate')
        ax.set_ylabel('TPR: True positive rate')
        ax.set_title('ROC Curve (area = {:.4f})'.format(auc))
        ax.grid()
        ax.plot(fpr, tpr)  # , marker='o')
        plt.savefig(os.path.join(os.getcwd(), 'training-roc.png'), transparent=True)
        plt.clf()
        # }}}

        # Save anomaly scores. {{{
        # =====
        df = pd.DataFrame({'anomaly_score': anomaly_scores,
                           'anomaly_label': anomaly_labels,
                           'labels': targets})
        df.to_csv(os.path.join(self.test_output_dir, 'training.anomaly_score.csv'), index=None)
        # }}}

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2)
        )
        return optimiser

    def _compute_anomaly_score(self, img, output):
        # Compute L1 loss
        res_loss = torch.abs(img - output)
        # res_loss = res_loss.view(res_loss.size()[0], -1)
        res_loss = res_loss.view(res_loss.size()[0], -1).detach().cpu().numpy()
        # res_loss = torch.mean(res_loss, 1)
        res_loss = np.mean(res_loss, 1)
        # res_loss = np.max(res_loss, 1)
        res_loss = torch.from_numpy(res_loss)

        return res_loss

    def _vis_recon_image(self, original, input, output, mask, score, index):
        original_img = original.detach().cpu().numpy().transpose(1, 2, 0).squeeze() * 0.5 + 0.5
        # input_img = input.detach().cpu().numpy().transpose(1, 2, 0).squeeze() * 0.5 + 0.5
        output_img = output.detach().cpu().numpy().transpose(1, 2, 0).squeeze() * 0.5 + 0.5
        # mask_img = mask.detach().cpu().numpy().transpose(1, 2, 0).squeeze() * 0.5 + 0.5
        dif_img = np.abs(original_img - output_img) if original_img.ndim == 3 else \
            np.expand_dims(np.abs(original_img - output_img), axis=2)
        score = score.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 3, squeeze=False)
        ax[0][0].imshow(original_img)
        ax[0][0].set_xticklabels([])
        ax[0][0].set_yticklabels([])
        ax[0][0].set_title('Input image')

        ax[0][1].imshow(np.clip(output_img, a_min=0, a_max=1))
        ax[0][1].set_xticklabels([])
        ax[0][1].set_yticklabels([])
        ax[0][1].set_title('Output image')

        im2 = ax[0][2].imshow(np.clip(np.linalg.norm(dif_img, ord=2, axis=2), a_min=0, a_max=1), cmap='cividis',
                              vmax=1, vmin=0)
        ax[0][2].set_xticklabels([])
        ax[0][2].set_yticklabels([])
        ax[0][2].set_title('Difference image')
        add_colorbar(im2)

        fig.subplots('AnomalyScore: {:.3f}'.format(score))
        plt.savefig(os.path.join(self.test_output_dir, 'training-{}.png'.format(index)),
                    transparent=True)
        plt.clf()
        plt.close()
