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
        E_t = np.transpose(E, (0,2,1))
        F_t = np.transpose(F, (1,0))
        # TODO: Mask A and A_hat
        A_diag_zero = self.zero_mask_diag(A)
        A_hat_diag_zero = self.zero_mask_diag(A_hat)
        A_hat_diag = self.zero_mask_diag(A, inverse=True)
        S1 = np.matmul(E_t, E_hat)
        print(S1.shape)
        S2 = A_diag_zero @ A_hat_diag_zero @ A_hat_diag @ A_hat_diag

        S21 = np. transpose(np.matmul(F_t, F_hat))

        S3 = np.matmul(S21, A_hat_diag)
        S = np.matmul(S1, S2) + S3
        
        return S

    def similarity_loop(self, A, A_hat, E, E_hat, F, F_hat):
        # We are going to itterate over pairs of (a,b) and (i,j)
        # np.nindex is oging to make touples to avoid two extra loops.
        ij_pairs = list(np.ndindex(A.shape))
        ab_pairs = list(np.ndindex(A_hat.shape))
        n = A.shape[0]
        k = A_hat.shape[0]

        # create en empty Similarity martrix.
        S = np.empty((n,n,k,k))

        # Now we start filling in the S matrix.
        for (i, j )in ij_pairs:
            for (a, b) in ab_pairs:
                # OMG this loop feels sooo wrong!
                if a != b and i != j:
                    A_scalar = A[i,j] * A_hat[a,b] * A_hat[a,a] * A_hat[b,b]
                    S[i,j,a,b] = np.matmul(np.transpose(E[i,j,:]), E_hat[a,b,:]) * A_scalar
                    del A_scalar
                elif a == b and i == j:
                    S[i,j,a,b] = np.matmul(np.transpose(F[i,:]), F_hat[a,:]) * A_hat[a,a]
                else:
                    # For some reason the similarity beteen two nodes for the case when one node is on the diagonal is not defined.
                    # We will set these points to zero unitl we find an better solution. 
                    S[i,j,a,b] = 0.
        return S



if __name__ == "__main__":

    # unnesseccary generation of a lollipop graph
    lollipop = nx.lollipop_graph(3, 3)
    lollipop.add_edge(3,2)



    # Let's define some dimensions :)
    n = 6
    k = 6
    d_e = 2
    d_n = 3

    # Generation of random test graphs
    A = np.ones((n,n))
    E = np.random.randint(2, size=(n,n,d_e))
    F = np.random.randint(2, size=(n,d_n))
    A_hat = np.random.randint(2, size=(k,k))
    E_hat = np.random.randint(2, size=(k,k,d_e))
    F_hat = np.random.randint(2, size=(k,d_n))

    # Test the class, acctually this should go in a test function and folder. Later...
    mpgm = MPGM()

    S = mpgm.similarity_loop(A, A_hat, E, E_hat, F, F_hat)
    print(S)
    print(S.shape)
    