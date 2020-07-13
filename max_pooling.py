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

    def affinity_loop(self, A, A_hat, E, E_hat, F, F_hat):
        # We are going to itterate over pairs of (a,b) and (i,j)
        # np.nindex is oging to make touples to avoid two extra loops.
        ij_pairs = list(np.ndindex(A.shape))
        ab_pairs = list(np.ndindex(A_hat.shape))
        n = A.shape[0]
        self.n = n
        k = A_hat.shape[0]
        self.k = k
        # create en empty affinity martrix.
        S = np.empty((n,n,k,k))

        # Now we start filling in the S matrix.
        for (i, j) in ij_pairs:
            for (a, b) in ab_pairs:
                # OMG this loop feels sooo wrong!
                if a != b and i != j:
                    A_scalar = A[i,j] * A_hat[a,b] * A_hat[a,a] * A_hat[b,b]
                    S[i,j,a,b] = np.matmul(np.transpose(E[i,j,:]), E_hat[a,b,:]) * A_scalar
                    del A_scalar
                elif a == b and i == j:
                    # Is it necessary to transpose? I am in doubt if numpy auto matches dimensions. Update: No, does not, stop being paranoid!!!
                    S[i,j,a,b] = np.matmul(np.transpose(F[i,:]), F_hat[a,:]) * A_hat[a,a]
                else:
                    # For some reason the similarity between two nodes for the case when one node is on the diagonal is not defined.
                    # We will set these points to zero until we find an better solution. 
                    S[i,j,a,b] = 0.
        return S
    
    def max_pool(self, S, n_iterations: int=6):
        """
        Input: Affinity matrix
        Output: Soft assignment matrix
        Args:
            S (np.array): Float affinity matrix of size (k,k,n,n)
            n_iterations (int): Number of iterations for calculating X
        """
        # The magic happens here, we are going to iteratively max pool the S matrix to get the X matrix.
        # We initiate the X matrix random uniform.
        # init X
        k = self.k
        n = self.n
        X = np.random.uniform(size=(n,k))
        # make pairs
        ia_pairs = list(np.ndindex(X.shape))

        #Just to make sure we are not twisting things. note: shape = dim+1
        assert ia_pairs[-1] == (n-1,k-1), 'Dimensions should be ({},{}) but are {}'.format(n-1,k-1,ia_pairs[-1])

        #loop over iterations and paris
        for itt in range(n_iterations):
            for (i, a) in ia_pairs:
                # TODO the paper says argmax and sum over the 'neighbors' of node pair (i,a).
                # My interpretation is that when there is no neighbor the S matrix will be zero, there fore we still use j anb b in full rage.
                # Second option would be to use a range of [i-1,i+2].
                # The first term max pools over the pairs of edge matches (ia;jb).
                de_sum = np.sum([np.argmax(X[j,:] @ S[i,j,a,:]) for j in range(k)])
                # In the next term we only consider the node matches (ia;ia).
                X[i,a] = X[i,a] * S[i,i,a,a] + de_sum
            # Normalize X to range [0,1].
            print(np.linalg.norm(X))
            X = X * 1./np.linalg.norm(X)
            print(X)

        return X




if __name__ == "__main__":

    # unnecessary generation of a lollipop graph
    lollipop = nx.lollipop_graph(3, 3)
    lollipop.add_edge(3,2)



    # Let's define some dimensions :)
    n = 6
    k = 2
    d_e = 2
    d_n = 3

    # Generation of random test graphs
    A = np.ones((n,n))
    E = np.random.randint(2, size=(n,n,d_e))
    F = np.random.randint(2, size=(n,d_n))
    A_hat = np.random.randint(2, size=(k,k))
    # A_hat = A TODO make it a distribution!!!
    E_hat = np.random.randint(2, size=(k,k,d_e))
    F_hat = np.random.randint(2, size=(k,d_n))

    # Test the class, acctually this should go in a test function and folder. Later...
    mpgm = MPGM()

    S = mpgm.affinity_loop(A, A_hat, E, E_hat, F, F_hat)
    print(S.shape)
    X = mpgm.max_pool(S)
