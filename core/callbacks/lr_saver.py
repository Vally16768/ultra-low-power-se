from __future__ import annotations
from tensorflow import keras
import tensorflow as tf

class LRSaver(keras.callbacks.Callback):
    """Pune learning_rate în logs, ca să apară în history.history."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            lr = self.model.optimizer.learning_rate
            lr_val = float(tf.keras.backend.get_value(lr))
        except Exception:
            lr_val = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        logs["learning_rate"] = lr_val
