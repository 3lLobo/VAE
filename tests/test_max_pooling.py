"""
Test the max-pooling alorithm.
"""

import networkx as nx
import numpy as np
from numpy import array
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
from max_pooling import MPGM
from munkres import Munkres, print_matrix, make_cost_matrix


class test_MPGM():
    def __init__(self):
        pass

    # def call_loop(self, A, A_hat, E, E_hat, F, F_hat):
    #     """
    #     A test run, does not work with batches. 1 to 1 implementation of the paper.
    #     Use this to verify your results if you decide to play around with the batch code.
    #     """
    #     S = self.affinity_loop(A, A_hat, E, E_hat, F, F_hat)
    #     X_star = self.max_pool_loop(S)
    #     X = self.hungarian(X_star)
    #     return X

    def test_call(self, A, A_hat, E, E_hat, F, F_hat):
        """
        Compare the results to the hard coded implementation.
        """
        mpgm = MPGM()
        test_S = mpgm.affinity(A, A_hat, E, E_hat, F, F_hat)
        S = self.affinity_loop(A, A_hat, E, E_hat, F, F_hat)

        assert test_S == S:
        
        #TODO keep doing thisss
        X_star = self.max_pool(S)
        X = self.hungarian_batch(X_star)
        return X)