import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
import tensorflow_probability as tfp
import numpy as np


class VanillaGVAE(Model):
    def __init__(self, n: int, na: int, ea: int, h_dim: int=512, z_dim: int=2):
        """
        Graph Variational Auto Encoder
        Args:
            n : Number of nodes
            na : Number of node attribues
            ea : Number of edge attributes
            h_dim : Hidden dimension
            z_dim : latent dimension
        """
        super().__init__()
        self.n = n
        self.na = na
        self.ea = ea

        self.encoder = tf.keras.Sequential(
            [
                Input(shape=[n*n + n*na + n*n*ea]),
                Dense(units=h_dim, activation='relu'),
                Dense(units=z_dim, ),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Input(shape=[z_dim]),
                Dense(units=h_dim, activation='relu'),
                Dense(units=(n*n + n*na + n*n*ea), activation='relu'),
            ]
        )
        
    def encode(self, A, E, F):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        Args:
            A: Adjancency matrix of size n*n
            E: Edge attribute matrix of size n*n*ea
            F: Node attribute matrix of size n*na
        """
        a = Flatten()(A)
        e = Flatten()(E)
        f = Flatten()(F)
        x = tf.concat([a, e, f], axis=1)
        print(x.shape())
        mean, logstd = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logstd
        
    def decode(self, z):
        logits = self.decoder(z)
        delimit_a = self.n*self.n
        delimit_e = self.n*self.n + self.n*self.n*self.ea

        a, e, f = logits[:,:delimit_a], logits[:,delimit_a:delimit_e], logits[:, delimit_e:]
        A = Reshape(target_shape=[self.n, self.n])(a)
        E = Reshape(target_shape=[self.n, self.n, self.ea])(e)
        F = Reshape(target_shape=[self.n, self.na])(f)
        return A, E, F
        
    def reparameterize(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logstd) + mean


if __name__ == "__main__":
    n = 20
    ea = 5
    na = 3

    A = np.random.randint((n, n))
    E = np.random.randint((n, n, ea))
    F = np.random.randint((n, na))
    model = VanillaGVAE(n, na, ea)

    mean, logstd = model.encode(A, E, F)
    z = model.reparameterize(mean, logstd)
    print('z', z)
    c = model.decode(z)
    print('Results:', Reshape(target_shape=[self.n, self.n])(a))
