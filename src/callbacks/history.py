import os

from pytorch_lightning.callbacks import Callback
from src.utils import History


class HistoryCallback(Callback):
    def __init__(self, output_dir=os.getcwd()):
        self.val_history = History(keys=('val_loss', '_'), output_dir=output_dir)
        self.train_history = History(keys=('train_loss', '_'), output_dir=output_dir)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for loss in outputs:
            self.train_history({'train_loss': loss.item()})

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for loss in outputs:
            self.train_history({'val_loss': loss.item()})

    def on_train_end(self, trainer, pl_module):
        self.train_history.plot_graph(filename='traininig_loss.png')
        self.train_history.save(filename='training_history.pkl')

    def on_validation_end(self, trainer, pl_module):
        self.val_history.plot_graph(filename='traininig_loss.png')
        self.val_history.save(filename='training_history.pkl')

