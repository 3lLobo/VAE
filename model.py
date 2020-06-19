import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, Reshape
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
            [
                Input(shape=[input_dim, input_dim, self.channels]),
                Conv2D(base_depth*2, 3, activation='relu'),
                Conv2D(base_depth, 3, activation='relu'),
                Flatten(),
                Dense(hidden_dim, activation='relu'),
                Dense(2*latent_dim),
            ]
        )

        # Define the decoder
        self.decoder = tf.keras.Sequential(
            [
                Input(shape=(latent_dim)),
                Dense(hidden_dim),
                Dense(18432, activation='relu'),
                Reshape(target_shape=(24, 24, 32)),
                Conv2DTranspose(base_depth, 3, activation='relu'),
                Conv2DTranspose(base_depth*2, 3, activation='relu'),
                Conv2DTranspose(self.channels, 3, padding='same', activation=None),
            ]
        )



    def encode(self, x):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        """
        x = tf.reshape(x, (-1, 28,28, 1))
        mean, logstd = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logstd
        
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mean, logstd):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logstd) + mean


model = VanillaVAE(28, 128, 8)

my_z = np.random.rand(1,28,28)

mean, logstd = model.encode(my_z)

my_lat = model.reparameterize(mean, logstd)
print('latent:', my_lat)
x_hat = model.decode(my_lat)
