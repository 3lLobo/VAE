import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
import tensorflow_probability as tfp
import numpy as np
from max_pooling import MPGM


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
                Dense(units=h_dim*2, activation='relu'),
                Dense(units=z_dim*2, ),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Input(shape=[z_dim]),
                Dense(units=h_dim*2, activation='relu'),
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
        a = tf.reshape(A, (-1, self.n*self.n))
        e = tf.reshape(E, (-1, self.n*self.n*self.ea))
        f = tf.reshape(F, (-1, self.n*self.na))
        x = tf.concat([a, e, f], axis=1)
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


def graph_loss(A, E, F, A_hat, E_hat, F_hat):
    """
    Loss function for the predicted graph. It takes each matrix separately into account.
    Goal is to solve the permutation invariance.
    Args:
        A_hat: Predicted adjencency matrix.
        E_hat: Predicted edge-attribute matrix.
        F_hat: Predicted node-attribute matrix.
        A: Ground truth adjencency matrix.
        E: Ground truth edge-attribute matrix.
        F: Ground truth node-attribute matrix.
    """
    # Set weights for diffenrent parts of the loss function
    w1 = 0
    w2 = 2
    w3 = 2
    w4 = 1

    # Match number of nodes
    loss_n_nodes = tf.math.sqrt(tf.cast(tf.math.count_nonzero(A) - tf.math.count_nonzero(A_hat), float)**2)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss = w1*loss_n_nodes + w2*bce(A, A_hat) + w3*bce(E, E_hat) + w4*bce(F, F_hat)
    return loss


def mpgm_loss(A, E, F, A_hat, E_hat, F_hat):
    """
    Loss function using max-pooling graph matching as describes in the GraphVAE paper.
    Lets see if backprop works. Args obvly the same as above!
    """
    n = A.shape[0]
    k = A_hat.shape[0]
    mpgm = MPGM()
    X = mpgm.call(A, A_hat, E, E_hat, F, F_hat)

    # now comes the loss part from the paper:
    A_t = X@A@X.T
    # or:
    A_t = tf.matmul(tf.matmul(X, A), X.T)
    E_hat_t = tf.matmul(tf.matmul(X, E_hat), X.T)
    F_hat_t = tf.matmul(X, F_hat)
    log_p_A = 1/k 
    pass





if __name__ == "__main__":
    n = 5
    ea = 5
    na = 3
    np.random.seed(seed=11)
    epochs = 111
    batch_size = 6
    
    model = VanillaGVAE(n, na, ea, h_dim=1024)
    optimizer = tf.optimizers.Adam(learning_rate=5e-4)
    A = np.random.randint(2, size=(batch_size, n, n))
    E = np.random.randint(2, size=(batch_size, n, n, ea))
    F = np.random.randint(2, size=(batch_size, n, na))    
    for epoch in range(epochs):
        # loss.backward
        with tf.GradientTape() as tape:
            mean, logstd = model.encode(A, E, F)
            z = model.reparameterize(mean, logstd)
            A_hat, E_hat, F_hat = model.decode(z)
            loss = graph_loss(A, E, F, A_hat, E_hat, F_hat)
            mpgm_loss(A, E, F, A_hat, E_hat, F_hat)
            print(loss.numpy())

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(A)
    A_hat = np.array(A_hat)
    A_hat[A_hat>.5] = 1.
    A_hat[A_hat<=.5] = 0.
    print('Pred:', A_hat)