import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np

class VanillaVAE(Model):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        To initialize the model define its dimensionalities
        Args:
            @hidden_dim: number of hidden dimensions
            @latent_dim: number of latent space dimensions
        """
        super().__init__()

        self.e1 = Conv2D(input_dim, 3, activation='relu')
        self.flatten = Flatten()
        self.e2 = Dense(hidden_dim, activation='relu')
        self.e3 = Dense(latent_dim*2)


    def encoder(self, x):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        """
        x = tf.compat.v1.expand_dims(x, -1)
        x = tf.cast(x, 'float32')
        x = self.coe1nv1(x)
        x = self.flatten(x)
        x = self.e2(x)
        x = self.e3(x)
        mu, log_std = x[:len(x)//2], x[len(x)//2:]
        return mu, log_std
        
    def decoder(self, x)

