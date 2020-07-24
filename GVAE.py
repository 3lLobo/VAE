import tensorflow as tf
from tensorflow.linalg import set_diag, diag_part
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.backend import batch_dot
import tensorflow_probability as tfp
import numpy as np
import time
from max_pooling import MPGM
from utils import mk_random_graph_ds


class VanillaGVAE(Model):
    def __init__(self, n: int, ea: int, na: int, h_dim: int=512, z_dim: int=2):
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
        
    def encode(self, args_in):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        Args:
            A: Adjancency matrix of size n*n
            E: Edge attribute matrix of size n*n*ea
            F: Node attribute matrix of size n*na
        """
        (A, E, F) = args_in
        a = tf.reshape(A, (-1, self.n*self.n))
        e = tf.reshape(E, (-1, self.n*self.n*self.ea))
        f = tf.reshape(F, (-1, self.n*self.na))
        x = tf.concat([a, e, f], axis=1)
        mean, logstd = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logstd
        
    def decode(self, z):
        logits = self.decoder(z)
        logits = tf.cast(logits, dtype=tf.float64)
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
        eps = tf.cast(tf.random.normal(shape=mean.shape), dtype=tf.float64)
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


def mpgm_loss(target, prediction, l_A=1., l_E=1., l_F=1.):
    """
    Loss function using max-pooling graph matching as describes in the GraphVAE paper.
    Lets see if backprop works. Args obvly the same as above!
    """
    A, E, F = target
    A_hat, E_hat, F_hat = prediction
    n = A.shape[1]
    k = A_hat.shape[1]
    mpgm = MPGM()
    X = tf.cast(mpgm.call(A, A_hat, E, E_hat, F, F_hat), dtype=tf.float64)

    # now comes the loss part from the paper:
    A_t = tf.transpose(X, perm=[0,2,1]) @ A @ X     # shape (bs,k,n)
    E_hat_t = tf.transpose(batch_dot(batch_dot(X, E_hat, axes=(-1,1)), X, axes=(-2,1)), perm=[0,1,3,2])
    F_hat_t = tf.matmul(X, F_hat)
    term_1 = (1/k) * tf.math.reduce_sum(diag_part(A_t) * tf.math.log(diag_part(A_hat)), [1], keepdims=True)
    term_2 = tf.reduce_sum((tf.ones_like(diag_part(A_t)) - diag_part(A_t)) * (tf.math.log(tf.ones_like(diag_part(A_hat)) - diag_part(A_hat))), [1], keepdims=True) 
    
    # TODO unsure if (1/(k*(1-k))) or ((1-k)/k) ??? Also the second sum in the paper is confusing. I am going to interpret it as matrix multiplication and sum over all elements.
    b = diag_part(A_t)
    term_31 = tf.matmul(set_diag(A_t, tf.zeros_like(diag_part(A_t))), tf.math.log(set_diag(A_hat, tf.zeros_like(diag_part(A_hat)))), transpose_a=True)
    term_32 = tf.matmul(tf.ones_like(A_t) - set_diag(A_t, tf.zeros_like(diag_part(A_t))), tf.math.log(tf.ones_like(A_t) - set_diag(A_hat, tf.zeros_like(diag_part(A_hat)))), transpose_a=True)
    term_3 = (1/k*(1-k)) * tf.expand_dims(tf.math.reduce_sum(term_31 + term_32, [1,2]), -1)
    log_p_A = term_1 + term_2 + term_3

    # Man so many confusions: is the log over one or both Fs???
    F = tf.cast(F, dtype=tf.float64)
    A = tf.cast(A, dtype=tf.float64)
    E = tf.cast(E, dtype=tf.float64)
    log_p_F = (1/n) * tf.expand_dims(tf.math.reduce_sum(tf.math.log(F) @ F_hat_t, [1,2]), -1)

    log_p_E = tf.expand_dims((1/(tf.norm(A, ord='fro', axis=[-2,-1])-n)) * tf.math.reduce_sum(tf.math.log(E) * E_hat_t,  [1,2,3]), -1)

    loss = - l_A * log_p_A - l_F * log_p_F - l_E * log_p_E
    return loss





if __name__ == "__main__":

    #Dear TensorFlow,
    #What I always wanted to tell you:
    tf.keras.backend.set_floatx('float64')

    n = 3
    d_e = 5
    d_n = 3
    np.random.seed(seed=11)
    epochs = 111
    batch_size = 4

    train_set = mk_random_graph_ds(n, d_e, d_n, 400, batch_size=batch_size)
    test_set = mk_random_graph_ds(n, d_e, d_n, 100, batch_size=batch_size)

    model = VanillaGVAE(n, d_e, d_n, h_dim=1024)
    optimizer = tf.optimizers.Adam(learning_rate=5e-4)
   
    for epoch in range(epochs):
        # loss.backward
        start_time = time.time()
        for target in train_set:
            with tf.GradientTape() as tape:
                mean, logstd = model.encode(target)
                z = model.reparameterize(mean, logstd)
                prediction = model.decode(z)
                loss = mpgm_loss(target, prediction)
                print(loss.numpy())
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        end_time = time.time()
        mean_loss = tf.keras.metrics.Mean()
        for test_x in test_ds:
            mean, logstd = model.encode(target)
            z = model.reparameterize(mean, logstd)
            prediction = model.decode(z)
            loss = mpgm_loss(target, prediction)
            mean_loss(loss)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, mean_loss.result(), end_time - start_time))