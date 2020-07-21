"""
Implementation of the max-pooling graph matching algorithm.
"""
import networkx as nx
import numpy as np
from numpy import array
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
from munkres import Munkres, print_matrix, make_cost_matrix


class MPGM():
    def __init__(self):
        pass

    def call(self, A, A_hat, E, E_hat, F, F_hat):
        S = self.affinity(A, A_hat, E, E_hat, F, F_hat)
        X_star = self.max_pool(S)
        X = self.hungarian(X_star)
        return X

    
    # def zero_mask_diag(self, A, inverse=False):
    #     # creates mask to zero out the diagonal matrix
    #     # If inverse is ser to true it returns a matrix with only the diagonal values
    #     if inverse:
    #         zeros = np.zeros_like(A)
    #         np.fill_diagonal(zeros, np.diag(A))
    #         return zeros
    #     else:       
    #         np.fill_diagonal(A, 0) 
    #         return A
    
    def zero_diag(self, A, inverse=False):
        """
        Returns either a zero matrix with only the diagonal of the input or the input matrix with the diagonal as zeros.
        Input can be a np.array and only 2 dimensional.
        """
        if inverse:
            return tf.linalg.diag(A)
        else:       
            return tf.linalg.set_diag(A, np.zeros((A.shape[0],A.shape[1])))
        
    def zero_diag_higher_dim(self, S):
        """
        Returns the input matrix with only the diagonal elements on the lowest dimensions (k,k).
        Input has to be of shape (batch_size,n,n,k,k).
        TODO mask the (n,n) diagonal elements as zero too!
        """
        batch_size, n, k = S.shape[0], S.shape[1], S.shape[3]
        S = tf.reshape(S, [-1,k,k])
        S = tf.linalg.set_diag(S, tf.zeros((S.shape[0],S.shape[1]), dtype=tf.float64))
        S = tf.reshape(S, [batch_size,n,n,k,k])
        return S

    def ident_matching_nk(bs, n, k):
        # Returns ... not sure anymore

        X = tf.zeros([bs, n, k])
        for i in range(min(k,n)):
            X[:,i,i] = 1
        return X

    def S2_matching_nnkk(self, S2, bs, n, k):
        """
        Returns zero matrix fof shape (bs,n,n,k,k) with the (n.n) and (k,k) diagonal set as in S2.
        Input is the S2 (bs,n,k) diagonals
        """

        X = np.zeros([bs,n,n,k,k])
        for i in range(n):
            tf_diag = tf.linalg.set_diag(tf.zeros([bs,k,k], dtype=tf.float64), S2[:,i,:])
            X[:,i,i,:,:] = tf_diag.numpy()
        print(X)
        return X

    def ident_matching_nnkk(self, bs, n, k):
        """
        Returns zero mask for (nn)  diagonal of a (bs,n,n,k,k) matrix.
        Input obvsl (bs,n,n,k,k)
        """

        X = np.ones([bs,n,n,k,k])
        for i in range(n):
            X[:,i,i,:,:] = 0
        return X

    def affinity(self, A, A_hat, E, E_hat, F, F_hat):
        """
        Let's make some dimensionalities clear first:
            A: nxn
            E: nxnxd_e
            F: nxd_n
            A_hat: kxk
            E_hat: kxkxd_e
            F_hat: kxd_n
        In an ideal world the target dimension n and the predictions dim k are the same.
        The other two dimensions are node and edge attributes. All matrixes come in batches, which is the first dimension.
        Now we are going to try to solve this with matrix multiplication, for-loops are a no-go.
        My first shot would be to implement this formula without respecting the constrains:
        S((i, j),(a, b)) = (E'(i,j,:)E_hat(a,b,:))A(i,j)A_hat(a,b)A_hat(a,a)A_hat(b,b) [i != j ∧ a != b] + (F'(i,:)F_hat(a,:))A_hat(a,a) [i == j ∧ a == b]
        And later mask the constrained entries with zeros.
        TODO To test it we could run a single sample and compare the loop and the matmul output.
        """
        n = A.shape[1]
        self.n = n
        k = A_hat.shape[1]
        self.k = k
        bs = A.shape[0]     # bs stands for batch size, just to clarify.
        self.bs = bs

        F_hat_t = tf.transpose(F_hat, perm=(0,2,1))
        A_hat_diag = tf.expand_dims(tf.linalg.diag_part(A_hat),-1)
        A_hat_diag_t = tf.transpose(A_hat_diag, perm=[0, 2, 1])

        # Cast the matices to tensors, bc the function tensordot only takes tensors..
        F = tf.cast(F, dtype=tf.float64)
        A = tf.cast(A, dtype=tf.float64)
        E = tf.cast(E, dtype=tf.float64)
        E_hat = tf.cast(E_hat, dtype=tf.float64) 
        S11 = tf.keras.backend.batch_dot(E, E_hat, axes=(3, 3))   # Crazy that this function even exists. We aim for shape (batch_s,n,n,k,k).

        # Now we need to get the second part into shape (batch_s,n,n,k,k).
        S121 = A_hat * (A_hat_diag @ A_hat_diag_t)
        # This step masks out the (a,b) diagonal. TODO: Make it optional.
        S122 = tf.linalg.set_diag(S121, tf.zeros((S121.shape[0],S121.shape[1]), dtype=tf.float64))
        S12 = tf.expand_dims(S122, -1)

        # This step masks out the (a,b) diagonal. TODO: Make it optional.
        S131 = tf.linalg.set_diag(A, tf.zeros((A.shape[0],A.shape[1]), dtype=tf.float64))
        A = tf.expand_dims(A, -1)
        S13 = tf.keras.backend.batch_dot(A, S12, axes=(-1,-1))
        
        # Pointwise multiplication of E and A part? Does this make sense?
        S1 = S11 * S13

        S21 = tf.tile(A_hat_diag, [1,1,n]) # This repeats the input vector to match the F shape.
        S2 = tf.matmul(F, F_hat_t) * tf.transpose(S21, perm=(0,2,1))     # I know this looks weird but trust me, I thought this through!
        # This puts the values on the intended diagonal to match the shape of S
        S2 = self.S2_matching_nnkk(S2, bs, n, k)

        # This zero masks the (n,n) diagonal
        S1 = S1 * self.ident_matching_nnkk(bs, n, k)

        return S1 + S2

    def affinity_loop(self, A, A_hat, E, E_hat, F, F_hat):
        # We are going to iterate over pairs of (a,b) and (i,j)
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
    
    def max_pool(self, S, n_iterations: int=30):
        """
        The famous Cho max pooling in matrix multiplication style.
        Xs: X_star meaning X in continous space.
        """
        # Just a crazy idea, but what if we falatten the X (n,k) matrix so that we can take the dot product with S (n,flat,K).
        Xs = tf.random.uniform(shape=[self.bs, self.n, self.k], dtype=tf.float64)
        S = tf.reshape(S, [S.shape[0],S.shape[1],-1,S.shape[-1]])
        for n in range(n_iterations):
            Xs = tf.reshape(Xs, [self.bs,-1])
            SXs = tf.keras.backend.batch_dot(S, Xs, axes=[2,1])
            Xs = SXs / tf.norm(SXs, ord='fro', axis=[-2,-1])[:,tf.newaxis,tf.newaxis]
        return Xs

    def max_pool_loop(self, S, n_iterations: int=6):
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
            print(X)
            X = X * 1./np.linalg.norm(X)
        return X

    def hungarian(self, X_star, cost: bool=True):
        
        """ 
        Apply the hungarian or munkes algorithm to the continuous assignment matrix.
        The output is a discrete similarity matrix.
        Are we working with a cost or a profit matrix???
        Args:
            X_star: numpy array matrix of size n x k with elements in range [0,1]
            cost: Boolean argument if to assume the input is a profit matrix and to convert is to a cost matrix or not.
        """
        m = Munkres()
        if cost:
            X_star = make_cost_matrix(X_star)
        # Compute the indexes for the matrix for the lowest cost path.        
        indexes = m.compute(X_star)
        print(indexes[0])

        # Now mast these indexes with 1 and the rest with 0.
        X = np.zeros_like(X_star, dtype=int)
        for idx in indexes:
            X[idx] = 1
        return X

    # def hungarian_batch(self, Xs):
    #     Xs = Xs.numpy()
    #     X = np.zeros((self.bs, self.k, self.k))
    #     for i in range(Xs.shape[0]):
    #         hung = np.expand_dims(linear_sum_assignment(np.squeeze(X[i,:,:])), axis=0)
    #         X[i,:,:] = hung
    #     return X

    def hungarian_batch(self, X):
        X = X.numpy()
        for i in range(X.shape[0]):
            # We are always given square Xs, but some may have unused columns (ground truth nodes are not there), so we can crop them for speedup. It's also then equivalent to the original non-batched version.
            row_ind, col_ind = linear_sum_assignment(X[i])
            M = np.zeros(X[i].shape, dtype=np.float32)
            M[row_ind, col_ind] = 1
            X[i] = M
        return X



if __name__ == "__main__":

    # Let's define some dimensions :)
    n = 3
    k = 3
    d_e = 2
    d_n = 3

    batch_size = 16

    # Generation of random test graphs. The target graph is discrete and the reproduced graph probabilistic.
    np.random.seed(seed=11)
    A = np.random.randint(2, size=(batch_size,n,n))
    E = np.random.randint(2, size=(batch_size,n,n,d_e))
    F = np.random.randint(2, size=(batch_size,n,d_n))
    A_hat = np.random.normal(size=(batch_size,k,k))
    E_hat = np.random.normal(size=(batch_size,k,k,d_e))
    F_hat = np.random.normal(size=(batch_size,k,d_n))


    # Test the class, actually this should go in a test function and folder. Later...
    mpgm = MPGM()

    S = mpgm.affinity(A, A_hat, E, E_hat, F, F_hat)
    print(S)
    Xs = mpgm.max_pool(S)
    X = mpgm.hungarian_batch(Xs)
    print(X)
    # X2 = mpgm.affinity_loop(np.squeeze(A), np.squeeze(A_hat), np.squeeze(E), np.squeeze(E_hat), np.squeeze(F), np.squeeze(F_hat))
    # print(X2)
