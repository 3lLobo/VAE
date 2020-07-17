import numpy as np
import tensorflow as tf


n = 3
k = 2
d_e = 2
d_n = 2

batch_size = 3
A = np.random.randint(2, size=(batch_size,n,n))
E = np.random.randint(2, size=(batch_size,n,n,d_e))
F = np.random.randint(2, size=(batch_size,n,d_n))
A_hat = np.random.normal(size=(batch_size,k,k))
E_hat = np.random.normal(size=(batch_size,k,k,d_e))
F_hat = np.random.normal(size=(batch_size,k,d_n))

def zero_mask_diag_tensor(A, inverse=False):
    """
    Decided to use tensors since numpy makes batch-life hard.
    Input can be a np.array?
    """
    if inverse:
        return tf.linalg.diag(A)
    else:    
        return tf.linalg.tensor_diag(tf.linalg.diag_part(A)) 
        # return tf.linalg.set_diag(A, tf.zeros((A.shape[0],A.shape[1],A.shape[2],A.shape[3]), dtype=tf.float64))

# print(tf.linalg.set_diag(A, np.zeros((A.shape[0],A.shape[1]))))
# print(tf.linalg.diag(A))

# E = tf.cast(E, dtype=tf.float32)
# E_hat = tf.cast(E_hat, dtype=tf.float32)    # The function tensordot only takes tensors.. surprise X)
# S1 = tf.keras.backend.batch_dot(E, E_hat, axes=(3, 3))
# S1 = tf.matmul(E, tf.transpose(E_hat, perm=[0, 3, 1, 2]))

# A_d = tf.expand_dims(tf.linalg.diag_part(A_hat), -1)
# print(A_d)
# A_dd = tf.tile(A_d, [1,1,n])

E = tf.cast(E, dtype=tf.float64)
E_hat = tf.cast(E_hat, dtype=tf.float64) 
S = tf.keras.backend.batch_dot(E, E_hat, axes=(3, 3))
S_c = tf.identity(S)
# S = tf.linalg.diag_part(S)
# S = tf.transpose(S, perm=[0,1,3,4,2])
# S = zero_mask_diag_tensor(S)
# S = tf.transpose(S, perm=[0,1,4,2,3])
S = tf.reshape(S, [-1,k,k])
S = tf.linalg.set_diag(S, tf.zeros((S.shape[0],S.shape[1]), dtype=tf.float64))
S = tf.reshape(S, [batch_size,n,n,k,k])
print(S,S_c)