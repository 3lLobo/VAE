import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose
import tensorflow_probability as tfp
import numpy as np

class VanillaVAE(Model):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, base_depth: int =32):
        """
        To initialize the model define its dimensionalities
        Args:
            @hidden_dim: number of hidden dimensions
            @latent_dim: number of latent space dimensions
            @base_depth: number of filters for convolution layers
        """
        super().__init__()
        self.channels = 1
        # Define the encoder
        self.encoder = tf.keras.Sequential(
            Input(shape=(input_dim, input_dim, self.channels)),
            Conv2D(base_depth*2, 3, activation='relu'),
            Conv2D(base_depth, 3, activation='relu'),
            Flatten(),
            Dense(hidden_dim, activation='relu'),
            Dense(latent_dim*2),
        )

        # Define the decoder
        self.decoder = tf.keras.Sequential(
            Input(shape=(latent_dim)),
            Dense(hidden_dim),
            Conv2DTranspose(base_depth, 3),
            Conv2DTranspose(base_depth*2, 3),
            Conv2D(self.channels, 3, activation=None),
        )



    def encoder(self, x):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        """
        x = tf.compat.v1.expand_dims(x, -1)
        x = tf.cast(x, 'float32')
        x = self.encoder(x)
        mu, log_std = x[:len(x)//2], x[len(x)//2:]
        return mu, log_std

    def decoder(self, z):
        return self.decoder(z)


model = VanillaVAE(28, 128, 8)

my_z = np.random.rand((1,8))
print(my_z)
x_hat = model.decoder(my_z)
print(x_hat)