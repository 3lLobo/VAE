"""
Implementation of the max-pooling graph matching algorithm.
"""
import networkx as nx
import numpy as np


class MPGM():
    def __init__(self):
        pass

    def call(self, A, A_hat, E, E_hat, F, F_hat):
        S = self.similarity(A, A_hat, E, E_hat, F, F_hat)
        pass
    
    def zero_mask_diag(self, A, inverse=False):
        # creates mask to zero out the diagonal matrix
        # If inverse is ser to true it returns a matrix with only the diagonal values
        if inverse:
            zeros = np.zeros_like(A)
            np.fill_diagonal(zeros, np.diag(A))
            return zeros
        else:       
            np.fill_diagonal(A, 0) 
            return A
        
    def similarity(self, A, A_hat, E, E_hat, F, F_hat):
        # Matrix multiplication possible?
        E_t = np.transpose(E, (2,0,1))
        F_t = np.transpose(F, (1,0))
        # TODO: Mask A and A_hat
        A_diag_zero = self.zero_mask_diag(A)
        A_hat_diag_zero = self.zero_mask_diag(A_hat)
        A_hat_diag = self.zero_mask_diag(A, inverse=True)
        S1 = np.matmul(E_t, E_hat)
        S2 = A_diag_zero @ A_hat_diag_zero @ A_hat_diag @ A_hat_diag
        S3 = np.matmul(np.matmul(F_t, F_hat), A_hat_diag)
        S = np.matmul(S1, S2) + S3
        
        return S


if __name__ == "__main__":

    # unnesseccary generation of a lollipop graph
    lollipop = nx.lollipop_graph(3, 3)
    lollipop.add_edge(3,2)

    # Generation of random test graphs
    A = np.ones((6,6))
    E = np.random.randint(2, size=(6,6,2))
    F = np.random.randint(2, size=(6,3))
    A_hat = np.random.randint(2, size=(6,6))
    E_hat = np.random.randint(2, size=(6,6,2))
    F_hat = np.random.randint(2, size=(6,3))

    # Test the class, acctually this should go in a test function and folder. Later...
    mpgm = MPGM()

    S = mpgm.similarity(A, A_hat, E, E_hat, F, F_hat)
    print(S)
