import os

from pytorch_lightning.callbacks import Callback
from src.utils import History


class HistoryCallback(Callback):
    def __init__(self, output_dir=os.getcwd()):
        self.val_history = History(keys=('val_loss', '_'), output_dir=output_dir)
        self.train_history = History(keys=('train_loss', '_'), output_dir=output_dir)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # print(outputs)
        self.train_history({'train_loss': outputs[0][0]['train_loss'].item()})
        # print('\n',outputs,'\n')
            # print(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # print('\n',outputs,'\n')
        self.val_history({'val_loss': outputs.item()})
        # for loss in outputs:
            # self.train_history({'val_loss': loss.item()})
            # pass

    def on_train_end(self, trainer, pl_module):
        self.train_history.plot_graph(filename='traininig_loss.png')
        self.train_history.save(filename='training_history.pkl')
        self.val_history.plot_graph(filename='validation_loss.png')
        self.val_history.save(filename='validation_history.pkl')

    def on_validation_end(self, trainer, pl_module):
        pass

    def teardown(self, trainer, pl_module, stage: str):
        """Called when fit or test ends"""
        