import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, Reshape
import tensorflow_probability as tfp
import numpy as np

class VanillaCVAE(Model):
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
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1, seed=None)
        # Define the encoder
        self.encoder = tf.keras.Sequential(
            [
                Input(shape=[input_dim, input_dim, self.channels]),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                Flatten(),
                # No activation
                Dense(latent_dim + latent_dim),
            ]
        )

        # Define the decoder
        self.decoder = tf.keras.Sequential(
            [
                Input(shape=(latent_dim)),
                Dense(units=7*7*32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                # No activation
                Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
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
        logits = self.decoder(z)
        return logits
        
    def reparameterize(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logstd) + mean

    def plot_manifold(self, epoch,latent_dim=10, n=4, digit_size=28):
        """
        Plots a 2D maifold of the last two dimensions of the latent space.
        The remeining latent dimensions are sampled randomly l_dim-2 times.
        """
        l_remain = int(latent_dim -2)
        l_r_half = int(l_remain/2)
        print(l_remain)
        norm = tfp.distributions.Normal(0, 1)
        grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
        grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
        image_width = digit_size*n
        image_height = image_width
        image = np.zeros((image_height, image_width))
        manifold = np.zeros((image_width*2, (l_r_half)*image_height))
        for l in range(l_remain):
          for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
              other_dims = (np.random.rand(1,l_remain)-0.5)*2
              z = np.array([[xi, yi]])
              z = np.concatenate((other_dims, z), axis=1)
              x_decoded = self.decode(z)
              x_decoded = tf.sigmoid(x_decoded)
              digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
              image[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit.numpy()
          if l < l_r_half:
            manifold[0:image_width, l*image_height:(l+1)*image_height] = image
          else:
            manifold[image_width:2*image_width, (l-l_r_half)*image_height:(l-l_r_half+1)*image_height] = image 
        plt.figure(figsize=(25, 100))
        plt.imshow(manifold, cmap='Greys_r')
        plt.axis('Off')
        plt.savefig('manifold_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

