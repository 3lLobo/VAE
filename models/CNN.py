import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np


# vanilla CNN model: inputs an image and outputs a prediction.
class VanillaCNN(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(28, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = tf.compat.v1.expand_dims(x, -1)
        x = tf.cast(x, 'float32')
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

