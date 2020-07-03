"""
Implementation of the max-pooling graph matching algorithm.
"""

import numpy as np


class MPGM():
    def __init__(self)
    pass

    def call(self, A, A_hat, E, E_hat, F, F_hat):
        S = self.similarity(A, A_hat, E, E_hat, F, F_hat)
        pass
    
    def similarity(self, A, A_hat, E, E_hat, F, F_hat):
        # Matrix multiplication possible?
        E_t = np.transpose(E, (2,0,1))
        F_t = np.transpose(F, (1,0))
        # TODO: Mask A and A_hat

        S = np.matmul(np.matmul(E_t, E_hat)), (A @ A_hat @ A_hat @ A_hat) + np.matmul(np.matmul(F_t, F_hat), A_hat)
        pass
