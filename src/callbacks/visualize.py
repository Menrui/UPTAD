from pytorch_lightning.callbacks import Callback


class VisualizeCallback(Callback):
    def on_init_start(self, trainer):
        pass

    def on_batch_end(self, trainer, pl_module):
        pass

    def on_epoch_end(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        pass

    def on_test_end(self, trainer, pl_module):
        pass
