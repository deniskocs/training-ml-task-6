import tensorflow as tf
from drawer import Drawer


class TrainCallback(tf.keras.callbacks.Callback):

    def __init__(self, drawer: Drawer):
        super().__init__()
        self.errors = []

        self.drawer = drawer

    def on_epoch_end(self, epoch, logs=None):
        self.errors.append(logs['loss'])

        self.drawer.drawErrors(self.errors)
        self.drawer.flush()
