from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import scipy 
import collections
import torch.multiprocessing as mp

import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
import ecc
import datasets

# workaround for https://github.com/pytorch/pytorch/issues/2317  (until new pytorch is released)
def fixedmax(t, dim, keepdim=False):
    v,i = torch.max(t,dim,keepdim)
    if not keepdim and v.dim()==t.dim() and v.size(dim)==1:
        return v.squeeze(dim), i.squeeze(dim)
    return v, i

def fixedmin(t, dim, keepdim=False):
    v,i = torch.min(t,dim,keepdim)
    if not keepdim and v.dim()==t.dim() and v.size(dim)==1:
        return v.squeeze(dim), i.squeeze(dim)
    return v, i


# def affinity_general(predictions, targets, use_pred_probabilities, nodes_in_edge_aff, x_quad_boost, fn_preproc, fn_edge_match, fn_node_match):
#     """ Construct matching affinity matrix.
#     - predictions: list of predicted complete graphs. Adjacency matrix contains node logits on the diagonal and edge logits off diagonal.
#     - targets: list of graphs. Adjacency matrix has no self-loops.
#     - use_pred_probabilities: Edge/node probabilities are considered in matching by multiplying affinities with them."""
#     # Expects non-variable tensors
#
#     As = []
#     t_nodef, t_adjacency, t_edgef = targets
#     p_nodef, p_adjacency, p_edgef = predictions
#
#     if use_pred_probabilities: p_adjacency = nnf.sigmoid(p_adjacency).data
#     if nodes_in_edge_aff=='hackP': p_adjacency = 0.25 + 0.5*p_adjacency
#
#     t_n = t_nodef.size(1)
#     p_n = p_nodef.size(1)
#
#         # add adjacency for masking invalid node combinations later
#         t_edgef = torch.cat([t_edgef, t_adjacency], 3)
#         p_edgef = fn_preproc(p_edgef)
#         if use_pred_probabilities:
#             p_edgef = torch.cat([p_edgef, (1-torch.eye(p_n)).unsqueeze(0) * p_adjacency], 3)
#         else:
#             p_edgef = torch.cat([p_edgef, 1-torch.eye(p_n)], 2)
#         #p_edgef = torch.cat([p_edgef, p_adjacency, 1-torch.eye(p_n)], 2)
#
#         # EDGES (off-diag)
#         ij = p_edgef.repeat(t_n, t_n, 1)  # prediction is "inner index" in A, so repeats
#         idx = torch.arange(0,t_n).long().view(-1,1).expand(t_n,p_n).contiguous().view(-1)
#         ab = torch.index_select(t_edgef, 0, idx)
#         ab = torch.index_select(ab, 1, idx) # target is "outer index" in A, so dilates
#
#         A = fn_edge_match(ij, ab)
#         if nodes_in_edge_aff=='hackA': A = 0.25 + 0.5*A
#         A = A * ij[:,:,-1] * ab[:,:,-1] # match only to edges in ground truth graph (ab) and prevent self-loops (ij)
#         #if use_pred_probabilities:  A = A * ij[:,:,-2]
#         #A[ab[:,:,-1]==0] = -1
#         #A[ij[:,:,-1]==0] = -1
#
#         # NODES (diag)
#         ab_columns = fn_node_match(p_nodef, t_nodef)
#         if nodes_in_edge_aff=='hackA': ab_columns = 0.25 + 0.5*ab_columns
#         node_affinity = ab_columns.t().contiguous().view(-1) # diagonal is vectorized column-wise
#         #t_marked = t_nodef.view(-1,1,2).transpose(0,2) #assumes one-hot
#         #p_marked = p_nodef.view(-1,2,1).transpose(0,1) #assumes (log?)softmaxed
#         #node_affinity = torch.addbmm(t_nodef.new(1,1).fill_(0), p_marked, t_marked) # negative ClassNLLCriterion
#         #node_affinity = node_affinity.transpose(0,1).contiguous().view(-1) # diagonal is column-wise vectorized
#
#         # todo: mismatch to loss if we use probabilites and not log-crit ?
#
#         if nodes_in_edge_aff in ['ap','a','p']:
#             # downweights contribution of edges in general and makes them contain node information as well
#             if nodes_in_edge_aff=='ap':
#                 vec = node_affinity * torch.diag(p_adjacency).repeat(t_n)
#             elif nodes_in_edge_aff=='a':
#                 vec = node_affinity
#             elif nodes_in_edge_aff=='p':
#                 vec = torch.diag(p_adjacency).repeat(t_n)
#             A = (A * vec.view(-1,1)) * vec.view(1,-1)
#
#         idx = torch.arange(0,t_n*p_n).long()
#
#         if use_pred_probabilities:
#             A[idx,idx] = node_affinity * torch.diag(p_adjacency).repeat(t_n) + x_quad_boost
#         else:
#             A[idx,idx] = node_affinity + x_quad_boost  # x_quad_boost effect: adds "+ x_quad_boost * norm(X)" into Cho's maximization problem
#
#         As.append( (A,t_n,p_n) )
#
#     return As


def affinity_general(predictions, targets, use_pred_probabilities, nodes_in_edge_aff, x_quad_boost, fn_preproc, fn_edge_match, fn_node_match):
    # Variable in, Tensors out
    """ Construct matching affinity matrix.
    - predictions: list of predicted complete graphs. Adjacency matrix contains node logits on the diagonal and edge logits off diagonal.
    - targets: list of graphs. Adjacency matrix has no self-loops.
    - use_pred_probabilities: Edge/node probabilities are considered in matching by multiplying affinities with them."""

    def cast(t): return t.cuda() if p_nodef[0].is_cuda else t

    t_nodef, t_adjacency, t_edgef, t_nmask = [t.data for t in targets]
    p_nodef, p_adjacency, p_edgef = [p.data for p in predictions]

    if use_pred_probabilities: p_adjacency = nnf.sigmoid(p_adjacency).data
    if nodes_in_edge_aff=='hackP': p_adjacency = 0.25 + 0.5*p_adjacency

    bs = t_nodef.size(0)
    t_n = t_nodef.size(1)
    p_n = p_nodef.size(1)

    # add adjacency for masking invalid node combinations later
    t_edgef = torch.cat([t_edgef, t_adjacency.unsqueeze(3)], 3)
    p_edgef = fn_preproc(p_edgef)
    if use_pred_probabilities:
        p_edgef = torch.cat([p_edgef, (cast(1-torch.eye(p_n)).unsqueeze(0) * p_adjacency).unsqueeze(3)], 3)
    else:
        p_edgef = torch.cat([p_edgef, cast(1-torch.eye(p_n)).unsqueeze(0).expand_as(p_adjacency).unsqueeze(3)], 3)
    #p_edgef = torch.cat([p_edgef, p_adjacency, 1-torch.eye(p_n)], 2)

    # EDGES (off-diag)
    ij = p_edgef.repeat(1, t_n, t_n, 1)  # prediction is "inner index" in A, so repeats
    idx = cast(torch.arange(0,t_n).long().view(-1,1).expand(t_n,p_n).contiguous().view(-1))
    ab = torch.index_select(t_edgef, 1, idx)
    ab = torch.index_select(ab, 2, idx) # target is "outer index" in A, so dilates

    A = fn_edge_match(ij, ab)
    if nodes_in_edge_aff=='hackA': A = 0.25 + 0.5*A
    A = A * ij[:,:,:,-1] * ab[:,:,:,-1] # match only to edges in ground truth graph (ab) and prevent self-loops (ij)
    #if use_pred_probabilities:  A = A * ij[:,:,-2]
    #A[ab[:,:,-1]==0] = -1
    #A[ij[:,:,-1]==0] = -1

    # NODES (diag)
    ab_columns = fn_node_match(p_nodef, t_nodef)
    node_affinity = ab_columns.transpose(1,2).contiguous().view(bs,-1) # diagonal is vectorized column-wise
    #t_marked = t_nodef.view(-1,1,2).transpose(0,2) #assumes one-hot
    #p_marked = p_nodef.view(-1,2,1).transpose(0,1) #assumes (log?)softmaxed
    #node_affinity = torch.addbmm(t_nodef.new(1,1).fill_(0), p_marked, t_marked) # negative ClassNLLCriterion
    #node_affinity = node_affinity.transpose(0,1).contiguous().view(-1) # diagonal is column-wise vectorized

    # todo: mismatch to loss if we use probabilites and not log-crit ?

    diags_rep = p_adjacency.masked_select(cast(torch.eye(p_n).byte()).unsqueeze(0)).view(bs,-1).repeat(1,t_n)

    if nodes_in_edge_aff in ['ap','a','p']:
        # downweights contribution of edges in general and makes them contain node information as well
        if nodes_in_edge_aff=='ap':
            vec = node_affinity * diags_rep
        elif nodes_in_edge_aff=='a':
            vec = node_affinity
        elif nodes_in_edge_aff=='p':
            vec = diags_rep
        A = (A * vec.view(bs,-1,1)) * vec.view(bs,1,-1)

    if use_pred_probabilities:
        An = node_affinity * diags_rep + x_quad_boost
    else:
        An = node_affinity + x_quad_boost  # x_quad_boost effect: adds "+ x_quad_boost * norm(X)" into Cho's maximization problem

    An = (An.view(bs,t_n,p_n) * t_nmask.unsqueeze(2)).view(bs,-1) # no affinity to empty ground truth edges

    #print(torch.diag(A[0]).sum()) ==0
    return A, An, t_n, p_n


def affinity_general_Var(predictions, targets, use_pred_probabilities, nodes_in_edge_aff, x_quad_boost, fn_preproc, fn_edge_match, fn_node_match):
    # Variable in, Variable out
    """ Construct matching affinity matrix.
    - predictions: list of predicted complete graphs. Adjacency matrix contains node logits on the diagonal and edge logits off diagonal.
    - targets: list of graphs. Adjacency matrix has no self-loops.
    - use_pred_probabilities: Edge/node probabilities are considered in matching by multiplying affinities with them."""

    def cast(t): return t.cuda() if p_nodef[0].data.is_cuda else t

    t_nodef, t_adjacency, t_edgef, t_nmask = targets
    p_nodef, p_adjacency, p_edgef = predictions

    if use_pred_probabilities: p_adjacency = nnf.sigmoid(p_adjacency)
    if nodes_in_edge_aff=='hackP': p_adjacency = 0.25 + 0.5*p_adjacency

    bs = t_nodef.size(0)
    t_n = t_nodef.size(1)
    p_n = p_nodef.size(1)

    # add adjacency for masking invalid node combinations later
    t_edgef = torch.cat([t_edgef, t_adjacency.unsqueeze(3)], 3)
    p_edgef = fn_preproc(p_edgef)
    if use_pred_probabilities:
        p_edgef = torch.cat([p_edgef, (Variable(cast(1-torch.eye(p_n)).unsqueeze(0)) * p_adjacency).unsqueeze(3)], 3)
    else:
        p_edgef = torch.cat([p_edgef, Variable(cast(1-torch.eye(p_n)).unsqueeze(0).expand_as(p_adjacency)).unsqueeze(3)], 3)
    #p_edgef = torch.cat([p_edgef, p_adjacency, 1-torch.eye(p_n)], 2)

    # EDGES (off-diag)
    ij = p_edgef.repeat(1, t_n, t_n, 1)  # prediction is "inner index" in A, so repeats
    idx = cast(torch.arange(0,t_n).long().view(-1,1).expand(t_n,p_n).contiguous().view(-1))
    ab = torch.index_select(t_edgef, 1, idx)
    ab = torch.index_select(ab, 2, idx) # target is "outer index" in A, so dilates

    A = fn_edge_match(ij, ab)
    if nodes_in_edge_aff=='hackA': A = 0.25 + 0.5*A
    A = A * ij[:,:,:,-1] * ab[:,:,:,-1] # match only to edges in ground truth graph (ab) and prevent self-loops (ij)
    #if use_pred_probabilities:  A = A * ij[:,:,-2]
    #A[ab[:,:,-1]==0] = -1
    #A[ij[:,:,-1]==0] = -1

    # NODES (diag)
    ab_columns = fn_node_match(p_nodef, t_nodef)
    node_affinity = ab_columns.transpose(1,2).contiguous().view(bs,-1) # diagonal is vectorized column-wise
    #t_marked = t_nodef.view(-1,1,2).transpose(0,2) #assumes one-hot
    #p_marked = p_nodef.view(-1,2,1).transpose(0,1) #assumes (log?)softmaxed
    #node_affinity = torch.addbmm(t_nodef.new(1,1).fill_(0), p_marked, t_marked) # negative ClassNLLCriterion
    #node_affinity = node_affinity.transpose(0,1).contiguous().view(-1) # diagonal is column-wise vectorized

    # todo: mismatch to loss if we use probabilites and not log-crit ?

    diags_rep = Variable(p_adjacency.data.masked_select(cast(torch.eye(p_n).byte()).unsqueeze(0)).view(bs,-1).repeat(1,t_n))

    if nodes_in_edge_aff in ['ap','a','p']:
        # downweights contribution of edges in general and makes them contain node information as well
        if nodes_in_edge_aff=='ap':
            vec = node_affinity * diags_rep
        elif nodes_in_edge_aff=='a':
            vec = node_affinity
        elif nodes_in_edge_aff=='p':
            vec = diags_rep
        A = (A * vec.view(bs,-1,1)) * vec.view(bs,1,-1)

    if use_pred_probabilities:
        An = node_affinity * diags_rep + x_quad_boost
    else:
        An = node_affinity + x_quad_boost  # x_quad_boost effect: adds "+ x_quad_boost * norm(X)" into Cho's maximization problem

    An = (An.view(bs,t_n,p_n) * t_nmask.unsqueeze(2)).view(bs,-1) # no affinity to empty ground truth edges

    #print(torch.diag(A[0]).sum()) ==0
    return A, An, t_n, p_n



# def max_pooling_matching_FLIPPED(As, discretize=False, max_iters=300):
#     """as in [Cho14] BUT APPLIED TO WRONG GRAPHS """
#     THRES = 1e-11
#
#     Xs = []
#     for (A,t_n,p_n) in As:
#         n = A.size(0)
#         X_prev = X = torch.ones(n).div_(n)
#
#         diag = torch.diag(A)
#         didx = torch.arange(0,t_n*p_n).long()
#         A[didx,didx] = 0
#
#         for i in range(max_iters):
#
#             cmuls = X.view(1,-1).expand_as(A) * A
#             pooled = fixedmax(cmuls.view(-1,t_n,p_n),1)[0]   # max over all b \in N_i (unconnected vertices were already zeroed out in A in construction)
#             X_new = X * diag + pooled.sum(1)            # sum over all j \in N_a (it's a complete graph & self-loop zeroed out in A in construction and main diagonal removed here)
#             nrm = X_new.norm()
#             if nrm > 0: X_new = X_new.div_(nrm)
#
#             diff = min(torch.dist(X_new, X), torch.dist(X_new, X_prev)) # two past X to prevent oscillations (as in original impl)
#             #print(i, diff, X.view(t_n,p_n).t())
#             X_prev = X
#             X = X_new
#             if diff < THRES: break
#
#         A[didx,didx] = diag
#
#         X = X.view(t_n,p_n).t()
#         if discretize:
#             # [Cho14] "MPM itself does not prevent one-to-many nor many-to-one matches if they are well supported by max-pooled neighboring matches"
#             row_ind, col_ind = scipy.optimize.linear_sum_assignment(-X.numpy()) # so Hungarian for the rescue of 1-1 assignment
#             X = np.zeros((p_n,t_n), dtype=np.float32)
#             X[row_ind, col_ind] = 1
#             X = torch.from_numpy(X)
#             #v,i = X.max(0)
#             #X.fill_(0).scatter_(0,row_ind.view(1,-1),1)
#
#         Xs.append(X)
#
#     return Xs


# def max_pooling_matching(As, discretize=False, aggr='', max_iters=300):
#     """as in [Cho14]"""
#
#     if aggr=='':
#         # My original implementation where ab=ground truth and ij=prediction. But the algorithm is not symmetric and Cho uses ab for noisy graphs and ij for clean (see Fig2), so it has to be reversed, this use doesn't make sense!
#         return max_pooling_matching_FLIPPED(As, discretize, max_iters)
#
#     THRES = 1e-11
#
#     Xs = []
#     for (A,t_n,p_n) in As:
#         n = A.size(0)
#         X_prev = X = A.new(n).fill_(1./n)
#
#         # affinity_general() constructs A in wrong order of inner/outer dimensions, so we need to tranpose it here. Then, ab = outer = prediction.
#         A = A.view(t_n,p_n,t_n,p_n).transpose(0,1).transpose(2,3).contiguous().view(p_n*t_n,p_n*t_n)
#
#         diag = torch.diag(A)
#         didx = torch.arange(0,t_n*p_n).long()
#         A[didx,didx] = 0
#
#         for i in range(max_iters):
#
#             cmuls = X.view(1,-1).expand_as(A) * A
#
#             if aggr=='sum':
#                 pooled = cmuls.view(-1,p_n,t_n).sum(1)
#                 X_new = X * diag + pooled.sum(1)
#             elif aggr=='mean':
#                 pooled = cmuls.view(-1,p_n,t_n).sum(1) / (cmuls.view(-1,p_n,t_n).gt(0).sum(1).float() + 1e-10)
#                 X_new = X * diag + pooled.sum(1)
#             elif aggr=='max':
#                 pooled = fixedmax(cmuls.view(-1,p_n,t_n),1)[0] # max over all b \in N_a (unconnected vertices were already zeroed out in A in construction)
#                 X_new = X * diag + pooled.sum(1)          # sum over all j \in N_i (it's a complete graph & self-loop zeroed out in A in construction and main diagonal removed here)
#
#             nrm = X_new.norm()
#             if nrm > 0: X_new = X_new.div_(nrm)
#
#             diff = min(torch.dist(X_new, X), torch.dist(X_new, X_prev)) # two past X to prevent oscillations (as in original impl)
#             #print(i, diff, X.view(t_n,p_n).t())
#             X_prev = X
#             X = X_new
#             if diff < THRES: break
#
#         #print('matching reward', X.view(1,-1).mm(A).mm(X.view(-1,1)))
#         X = X.view(p_n,t_n)
#         A[didx,didx] = diag
#
#         if discretize:
#             # [Cho14] "MPM itself does not prevent one-to-many nor many-to-one matches if they are well supported by max-pooled neighboring matches"
#             row_ind, col_ind = scipy.optimize.linear_sum_assignment(-X.cpu().numpy()) # so Hungarian for the rescue of 1-1 assignment
#             X = np.zeros((p_n,t_n), dtype=np.float32)
#             X[row_ind, col_ind] = 1
#             X = torch.from_numpy(X)
#             #v,i = X.max(0)
#             #X.fill_(0).scatter_(0,row_ind.view(1,-1),1)
#
#         Xs.append(X)
#
#     return Xs

def max_pooling_matching(As, discretize, aggr, matching_loss='', max_iters=300, nworkers=0):
    # Tensor in, Variable out
    """as in [Cho14]."""

    def cast(t): return t.cuda() if A[0][0].is_cuda else t

    A, diag, t_n, p_n = As   # Assumes that the diagonals of A are 0

    bs, n = A.size(0), A.size(1)
    X = A.new(bs,n).fill_(1./n)

    # affinity_general() constructs A in wrong order of inner/outer dimensions, so we need to tranpose it here. Then, ab = outer = prediction.
    A = A.view(bs,t_n,p_n,t_n,p_n).transpose(1,2).transpose(3,4).contiguous().view(bs,p_n*t_n,p_n*t_n)
    diag = diag.view(bs,t_n,p_n).transpose(1,2).contiguous().view(bs,p_n*t_n)

    for i in range(max_iters):

        cmuls = X.view(bs,1,-1).expand_as(A) * A

        if aggr=='sum':
            pooled = cmuls.view(bs,-1,p_n,t_n).sum(2)
            X_new = X * diag + pooled.sum(2)
        elif aggr=='mean':
            # UPD: this is bullshit, the divisor is always constant (it's over complete graph) ---> it effectively just makes the diagonal more dominant! Though that makes actually 'sum' more comparable to 'max', because max also makes diagonal more dominant than sum (just because there are less terms)! But actually, it should be more proper to use affinity_quad_boost for this.
            # The authors even say: "the proposed algorithm only maximizes a lower bound of the ﬁrst-order approximation of Eq.(3) at each iteration"
            pooled = cmuls.view(bs,-1,p_n,t_n).sum(2) / (cmuls.view(bs,-1,p_n,t_n).gt(0).sum(2).float() + 1e-10)
            X_new = X * diag + pooled.sum(2)
        elif aggr=='diag':
            X_new = X * diag
        elif aggr=='smax':
            pooled = datasets.logsumexpnd(cmuls.view(bs,-1,p_n,t_n).transpose(2,3).contiguous(), alpha=1)
            X_new = X * diag + pooled.sum(2)
        elif aggr=='max' or aggr=='': #('': not exactly legacy behavior but whatever)
            pooled = fixedmax(cmuls.view(bs,-1,p_n,t_n),2)[0] # max over all b \in N_a (it's a complete graph & self-loop zeroed out in A in construction and main diagonal removed here)
            X_new = X * diag + pooled.sum(2)          # sum over all j \in N_i (unconnected vertices were already zeroed out in A in construction)

        X = X_new.div(torch.norm(X_new,p=2,dim=1,keepdim=True).add(1e-7))

    X = X.view(bs,p_n,t_n)

    if matching_loss=='match_discr_sinkhorn': #1704.02729
        for j in range(10):
            s1 = X.sum(2, keepdim=True)
            if torch.var(s1.data) < 1e-8:
                break
            X = X / s1
            X = X / X.sum(1, keepdim=True)

    if matching_loss.startswith('match_discr_p'):
        p = int(matching_loss[len('match_discr_p'):])
        X = X.pow(p)

    if matching_loss.startswith('match_discr_gumbel'):
        temp = float(matching_loss[len('match_discr_gumbel'):])
        X = datasets.gumbel_sm(X.transpose(1,2), temp).transpose(1,2)

    if discretize:
        # [Cho14] "MPM itself does not prevent one-to-many nor many-to-one matches if they are well supported by max-pooled neighboring matches"
        X = Variable(cast(hungary_mt(X.cpu(), nworkers))) # so Hungarian for the rescue of 1-1 assignment   (btw, there is also a gpu impl: https://gist.github.com/paclopes/eca8944c17f0407253a141224508a4df)
        #v,i = X.max(0)
        #X.fill_(0).scatter_(0,row_ind.view(1,-1),1)
    else:
        X = Variable(X)

    return X


def max_pooling_matching_Var(As, discretize, aggr, matching_loss='', max_iters=300, nworkers=0):
    # Variable in, Variable out
    """as in [Cho14]. Here we try to maximize some metric derived from *non-discretized* X."""

    def cast(t): return t.cuda() if A[0][0].data.is_cuda else t

    A, diag, t_n, p_n = As   # Assumes that the diagonals of A are 0

    bs, n = A.size(0), A.size(1)
    X = Variable(A.data.new(bs,n).fill_(1./n))

    # affinity_general() constructs A in wrong order of inner/outer dimensions, so we need to tranpose it here. Then, ab = outer = prediction.
    A = A.view(bs,t_n,p_n,t_n,p_n).transpose(1,2).transpose(3,4).contiguous().view(bs,p_n*t_n,p_n*t_n)
    diag = diag.view(bs,t_n,p_n).transpose(1,2).contiguous().view(bs,p_n*t_n)

    for i in range(max_iters):

        cmuls = X.view(bs,1,-1).expand_as(A) * A

        if aggr=='sum':
            pooled = cmuls.view(bs,-1,p_n,t_n).sum(2)
            X_new = X * diag + pooled.sum(2)
        elif aggr=='mean':
            pooled = cmuls.view(bs,-1,p_n,t_n).sum(2) / (cmuls.view(bs,-1,p_n,t_n).gt(0).sum(2).float() + 1e-10)
            X_new = X * diag + pooled.sum(2)
        elif aggr=='max' or aggr=='': #('': not exactly legacy behavior but whatever)
            pooled = fixedmax(cmuls.view(bs,-1,p_n,t_n),2)[0] # max over all b \in N_a (it's a complete graph & self-loop zeroed out in A in construction and main diagonal removed here)
            X_new = X * diag + pooled.sum(2)          # sum over all j \in N_i (unconnected vertices were already zeroed out in A in construction)

        X = X_new.div(X_new.norm(dim=1,keepdim=True).add(1e-7))

    X = X.view(bs,p_n,t_n)

    if matching_loss=='match_discr_sinkhorn': #1704.02729
        for j in range(10):
            s1 = X.sum(2, keepdim=True)
            if torch.var(s1.data) < 1e-8:
                break
            X = X / s1
            X = X / X.sum(1, keepdim=True)

    if matching_loss.startswith('match_discr_p'):
        p = int(matching_loss[len('match_discr_p'):])
        X = X.pow(p)

    if matching_loss.startswith('match_discr_gumbel'):
        temp = float(matching_loss[len('match_discr_gumbel'):])
        X = datasets.gumbel_sm(X.transpose(1,2), temp).transpose(1,2)

    if discretize:
        # [Cho14] "MPM itself does not prevent one-to-many nor many-to-one matches if they are well supported by max-pooled neighboring matches"
        X = Variable(cast(hungary_mt(X.data.cpu(), nworkers))) # so Hungarian for the rescue of 1-1 assignment   (btw, there is also a gpu impl: https://gist.github.com/paclopes/eca8944c17f0407253a141224508a4df)
        #v,i = X.max(0)
        #X.fill_(0).scatter_(0,row_ind.view(1,-1),1)

    return X


def hungary(X):
    X = X.numpy()
    for i in range(X.shape[0]):
        # We are always given square Xs, but some may have unused columns (ground truth nodes are not there), so we can crop them for speedup. It's also then equivalent to the original non-batched version.
        last_valid = int(np.flatnonzero(np.sum(X[i],0) > 1e-8)[-1])
        C = -X[i][:,:last_valid+1]
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)
        M = np.zeros(X[i].shape, dtype=np.float32)
        M[row_ind, col_ind] = 1
        X[i] = M
    return torch.from_numpy(X)


# the motivation is that the iterative matching alg is the slowest (especially hungarian)
# multiprocess system inspired by dataloader, nothing else works well / fast enough (mt.Pool or even python threading)
def _worker_loop(id, in_queue, out_queue, func, f_args):
    torch.set_num_threads(1)
    while True:
        A = in_queue.get()
        if A is None:
            out_queue.put(None)
            break
        if isinstance(A, tuple):
            X = func(A[0], A[1], *f_args)
        else:
            X = func(A, *f_args)
        out_queue.put((id, X))

class WorkerHolder(object):
    def __init__(self, N, func, f_args):
        self.in_qs = [mp.Queue() for i in range(N)]
        self.out_q = mp.Queue()
        for i in range(N):
            proc = mp.Process(target=_worker_loop, args=(i, self.in_qs[i], self.out_q, func, f_args))
            proc.daemon = True
            proc.start()

    def __del__(self):
        for q in self.in_qs:
            q.put(None)

workerHolder = None

# def max_pooling_matching_mt(As, nworkers=0, discretize=False, aggr='', max_iters=300):
#     """ Splits the batch of max_pooling_matching into workers """
#     if nworkers==0:
#         return max_pooling_matching(As, discretize, aggr, max_iters)
#     else:
#         global workerHolder
#         if workerHolder is None: workerHolder = WorkerHolder(nworkers, max_pooling_matching, (discretize, aggr, max_iters))
#
#         recv, Xs_discr = {}, []
#         for i, split in enumerate(np.array_split(range(len(As)), nworkers)):
#             workerHolder.in_qs[i].put([As[s] for s in split])
#         for i in range(nworkers):
#             id, result = workerHolder.out_q.get()
#             recv[id] = result
#         for i in range(nworkers): Xs_discr += recv[i]
#         return Xs_discr
#
# def affinity_and_match_mt(affinity_func, predictions, targets, args, nworkers=0):
#     """ Splits the batch of affinity + max_pooling_matching into workers (large A matrices stay in workers:) )"""
#     if nworkers==0:
#         return affinity_func(predictions, targets, args)
#     else:
#         global workerHolder
#         if workerHolder is None: workerHolder = WorkerHolder(nworkers, affinity_func, (args,))
#
#         recv, Xs_discr = {}, []
#         for i, split in enumerate(np.array_split(range(len(targets)), nworkers)):
#             chunk = ([p[split,...] for p in predictions], [targets[s] for s in split])
#             workerHolder.in_qs[i].put(chunk)
#         for i in range(nworkers):
#             id, result = workerHolder.out_q.get()
#             recv[id] = result
#         for i in range(nworkers): Xs_discr += recv[i]
#         return Xs_discr


def hungary_mt(Xs, nworkers=0):
    """ Splits the batch of max_pooling_matching into workers """
    if nworkers==0:
        return hungary(Xs)
    else:
        global workerHolder
        if workerHolder is None: workerHolder = WorkerHolder(nworkers, hungary, ())

        recv, Xs_discr = {}, []
        for i, split in enumerate(np.array_split(range(len(Xs)), nworkers)):
            workerHolder.in_qs[i].put(Xs[split,...])
        for i in range(nworkers):
            id, result = workerHolder.out_q.get()
            recv[id] = result
        for i in range(nworkers): Xs_discr.append(recv[i])
        return torch.cat(Xs_discr)


def direct_X_loss(As, Xs, matching_loss):
    Ls = []
    for (A,X) in zip(As, Xs):
        if matching_loss=='reward': # want high reward (the actual matching objective)
            L = - X.view(1,-1).mm(A[0]).mm(X.view(-1,1))
        elif matching_loss.startswith('l'):  # want high norm (L2...and towards Linf norm should promote only a few strong peaks?)
            p = int(matching_loss[len('l'):])
            L = - X.norm(p=p)
        elif matching_loss=='sm': # want strong matches for each gt node (doesn't handle multiple assignments to a same predicted node, though)
            L = - nnf.softmax(X.t()).max(1)[0].sum()
        elif matching_loss=='max':
            L = - X.max(0)[0].sum()

        Ls.append(L)

    return torch.cat(L).mean()


def matching_loss_general(predictions, targets, Xs, fn_node_loss, fn_edge_loss):
    # Expects all variables

    losses_node_prob, losses_edge_prob, losses_node_feat, losses_edge_feat = [],[],[],[]
    def var(t): return Variable(t.cuda()) if predictions[0][0].is_cuda else Variable(t)

    for s,X in enumerate(Xs):
        t_nodef, t_adjacency, t_edgef, t_nmask = targets[0][s], targets[1][s], targets[2][s], targets[3][s]         #t_adjacency - binary and without self-loops
        p_nodef, p_adjacency, p_edgef = predictions[0][s], predictions[1][s], predictions[2][s]     #p_adjacency - logits =  before sigmoid

        gtidx = t_nmask.data.nonzero().squeeze()
        t_nodef = t_nodef[gtidx,:]
        t_adjacency = t_adjacency[gtidx,:][:,gtidx]
        t_edgef = t_edgef[gtidx,:,:][:,gtidx,:]
        X = X[:,gtidx]

        # probabilities of nodes & edges
        target_node_prob = fixedmax(X,1)[0] # makes likely sense for discrete X only
        loss_node_prob = nnf.binary_cross_entropy_with_logits(torch.diag(p_adjacency), target_node_prob, size_average=True)

        t_adjacency_sel = X.mm(t_adjacency).mm(X.t())
        idx_nondiag = var(torch.nonzero((torch.eye(p_adjacency.size(0))-1).view(-1)).squeeze())
        loss_edge_prob = nnf.binary_cross_entropy_with_logits(p_adjacency.view(-1)[idx_nondiag], t_adjacency_sel.view(-1)[idx_nondiag], size_average=True)

        # features of nodes
        node_feat_idx = fixedmax(X,0)[1]
        matched_features = p_nodef.index_select(0,node_feat_idx) # constraining only on ground truth features
        loss_node_feat = fn_node_loss(matched_features, t_nodef)

        # features of edges
        p_edgef_sel = X.t().matmul(p_edgef.transpose(0,2)).matmul(X).transpose(0,2)
        #matched_diffs = (p_edgef_sel - t_edgef) * t_adjacency.unsqueeze(2) # constraining only on ground truth edges
        #loss_edge_feat = matched_diffs.pow(2).sum() / t_adjacency.sum()
        idx_gt = torch.nonzero(t_adjacency.data) # constraining only on ground truth edges
        if idx_gt.numel()>0:
            loss_edge_feat = fn_edge_loss(p_edgef_sel[idx_gt[:,0],idx_gt[:,1]], t_edgef[idx_gt[:,0],idx_gt[:,1]])
        else:
            loss_edge_feat = Variable(loss_node_prob.data.new(1).fill_(0))

        if False: # arbitrary desire to have large entropy of unmatched features ;)  ... code only for qm9
            node_feat_idx = X.data.sum(1).eq(0).nonzero().squeeze()
            if node_feat_idx.numel() > 0:
                unmatched_features = p_nodef.index_select(0,Variable(node_feat_idx))
                probs = datasets.softmaxnd(unmatched_features)
                loss_node_feat = loss_node_feat + (probs * torch.log(probs + 1e-8)).sum(1).mean(0)

            probs = datasets.softmaxnd(p_edgef)
            negH = (probs * torch.log(probs + 1e-8)).sum(2)
            neg_t_adj = 1 - t_adjacency_sel
            negH = (negH * neg_t_adj).sum().div(neg_t_adj.sum() + 1e-6)
            loss_edge_feat = loss_edge_feat + negH

        losses_node_prob.append(loss_node_prob)
        losses_edge_prob.append(loss_edge_prob)
        losses_node_feat.append(loss_node_feat)
        losses_edge_feat.append(loss_edge_feat)

    return torch.cat(losses_node_prob).mean(), torch.cat(losses_edge_prob).mean(), torch.cat(losses_node_feat).mean(), torch.cat(losses_edge_feat).mean()





def matching_accur_general(predictions, targets, Xs, fn_node_acc, fn_edge_acc):
    """ As matching_loss_general, but measures mean accuracies instead of loss computation """
    # Expects all variables

    acces_node_prob, acces_edge_prob, acces_node_feat, acces_edge_feat = [],[],[],[]
    def var(t): return Variable(t.cuda()) if predictions[0][0].is_cuda else Variable(t)

    for s,X in enumerate(Xs):
        t_nodef, t_adjacency, t_edgef, t_nmask = targets[0][s], targets[1][s], targets[2][s], targets[3][s]         #t_adjacency - binary and without self-loops
        p_nodef, p_adjacency, p_edgef = predictions[0][s], predictions[1][s], predictions[2][s]     #p_adjacency - logits =  before sigmoid

        gtidx = t_nmask.data.nonzero().squeeze()
        t_nodef = t_nodef[gtidx,:]
        t_adjacency = t_adjacency[gtidx,:][:,gtidx]
        t_edgef = t_edgef[gtidx,:,:][:,gtidx,:]
        X = X[:,gtidx]

        # probabilities of nodes & edges
        target_node_prob = fixedmax(X,1)[0] # makes likely sense for discrete X only
        acc_node_prob = torch.diag(p_adjacency).gt(0).float().eq(target_node_prob).float().mean()  # replace mean() calls with min() -> % of perfect prediction :)

        t_adjacency_sel = X.mm(t_adjacency).mm(X.t())
        idx_nondiag = var(torch.nonzero((torch.eye(p_adjacency.size(0))-1).view(-1)).squeeze())
        acc_edge_prob = p_adjacency.view(-1)[idx_nondiag].gt(0).float().eq(t_adjacency_sel.view(-1)[idx_nondiag]).float().mean()

        # features of nodes
        node_feat_idx = fixedmax(X,0)[1]
        matched_features = p_nodef.index_select(0,node_feat_idx) # constraining only on ground truth features
        acc_node_feat = fn_node_acc(matched_features, t_nodef)

        # features of edges
        p_edgef_sel = X.t().matmul(p_edgef.transpose(0,2)).matmul(X).transpose(0,2)
        #matched_diffs = (p_edgef_sel - t_edgef) * t_adjacency.unsqueeze(2) # constraining only on ground truth edges
        #loss_edge_feat = matched_diffs.pow(2).sum() / t_adjacency.sum()
        idx_gt = torch.nonzero(t_adjacency.data) # constraining only on ground truth edges
        if idx_gt.numel()>0:
            acc_edge_feat = fn_edge_acc(p_edgef_sel[idx_gt[:,0],idx_gt[:,1]], t_edgef[idx_gt[:,0],idx_gt[:,1]])
        else:
            acc_edge_feat = Variable(acc_node_prob.data.new(1).fill_(0))

        acces_node_prob.append(acc_node_prob.data.squeeze()[0])
        acces_edge_prob.append(acc_edge_prob.data.squeeze()[0])
        acces_node_feat.append(acc_node_feat.data.squeeze()[0])
        acces_edge_feat.append(acc_edge_feat.data.squeeze()[0])

    return acces_node_prob, acces_edge_prob, acces_node_feat, acces_edge_feat



Mcache = {}

def matching_cache(Xs, extrainfo):
    # The idea is to fix matchings when the network has roughly stabilized to finalize the convergence
    global Mcache
    res = []
    for X,ei in zip(Xs.data.cpu(),extrainfo):
        if ei in Mcache:
            #if (X - Mcache[ei]).abs().sum()>0: print('change', ei, (X - Mcache[ei]).abs().sum())
            pass
        else:
            Mcache[ei] = X.clone()
            #print('new entry', ei)
        res.append(Mcache[ei])
    res = torch.stack(res,0)
    return Variable(res.cuda() if Xs.is_cuda else res)

# def matching_cache(Xs, extrainfo):
#     # Just to confirm that changes happen and matching does not converge.
#     global Mcache
#     res = []
#     for X,ei in zip(Xs.data.cpu(),extrainfo):
#         if ei in Mcache:
#             if (X - Mcache[ei]).abs().sum()>0: print('change', ei, (X - Mcache[ei]).abs().sum())
#         Mcache[ei] = X.clone()
#     return Xs

def try_cache(extrainfo, is_cuda):
    global Mcache
    res = []
    for ei in extrainfo:
        if ei not in Mcache:
            return None
        res.append(Mcache[ei])
    res = torch.stack(res,0)
    return Variable(res.cuda() if is_cuda else res)



# # this doesn't work, IMHO the QP problem is good but the optimizer produces total shit. Maybe because it's a non-convex problem?
#
# from qpth.qp import QPFunction, QPSolvers #https://locuslab.github.io/qpth/
#
# def qp_matching_Var(As):
#     def cast(t): return t.cuda() if A[0][0].data.is_cuda else t
#
#     Xs = []
#     for (A,t_n,p_n) in As:
#         n = A.size(0)
#         # affinity_general() constructs A in wrong order of inner/outer dimensions, so we need to tranpose it here. Then, ab = outer = prediction.
#         Q = A.view(t_n,p_n,t_n,p_n).transpose(0,1).transpose(2,3).contiguous().view(p_n*t_n,p_n*t_n)
#
#         # G0 = cast(-torch.eye(n)) # non-negativity
#         # G1 = Q.data.new(p_n,p_n,t_n).fill_(0) # <=1 for nodes of predicted graph  (#nodes in predicted graph >= #nodes in gt graph)
#         # for k in range(p_n):
#         #     G1[k,k,:].fill_(1)  #todo: these could be done by expanded eye (similar for A)
#         # G = torch.cat([G0, G1.view(p_n,-1)],0)
#         #
#         # h = Q.data.new(n+p_n)
#         # h[:n].fill_(0)
#         # h[n:].fill_(1)
#         #
#         # A = Q.data.new(t_n,p_n,t_n).fill_(0) # == for nodes of target graph
#         # for k in range(t_n):
#         #     A[k,:,k].fill_(1)
#         # A = A.view(t_n,-1)
#         # b = Q.data.new(t_n).fill_(1)
#
#         # e0 = Variable(Q.data.new(n).fill_(0))
#         # X = QPFunction(maxIter=50)(-Q, e0, Variable(G), Variable(h), Variable(A), Variable(b)) #,solver=QPSolvers.CVXPY
#         # X = X.view(p_n,t_n)
#         # Xs.append(X)
#
#         G0 = cast(-torch.eye(n)) # non-negativity
#         G1 = Q.data.new(p_n,p_n,t_n).fill_(0) # <=1 for nodes of predicted graph  (#nodes in predicted graph >= #nodes in gt graph)
#         for k in range(p_n):
#             G1[k,k,:].fill_(1)  #todo: these could be done by expanded eye (similar for A)
#         G2 = Q.data.new(t_n,p_n,t_n).fill_(0) # == for nodes of target graph
#         for k in range(t_n):
#             G2[k,:,k].fill_(1)
#         G = torch.cat([G0, G1.view(p_n,-1), G2.view(t_n,-1)],0)
#
#         h = Q.data.new(n+p_n+t_n)
#         h[:n].fill_(0)
#         h[n:].fill_(1)
#
#         e = Variable(torch.Tensor())
#         e0 = Variable(Q.data.new(n).fill_(0))
#         X = QPFunction(maxIter=50)(-Q, e0, Variable(G), Variable(h), e, e) #
#         X = X.view(p_n,t_n)
#         Xs.append(X)
#
#     return Xs






def ident_matching(predictions, targets):
    # identity matrices as assignment
    t_nodef, t_adjacency, t_edgef, t_nmask = [t.data for t in targets]
    p_nodef, p_adjacency, p_edgef = [p.data for p in predictions]

    bs = t_nodef.size(0)
    t_n = t_nodef.size(1)
    p_n = p_nodef.size(1)
    X = p_nodef[0].new(bs,p_n,t_n).fill_(0)
    for i in range(min(t_n,p_n)):
        X[:,i,i] = 1
    return Variable(X)






import unittest


class TestGraphMatching(unittest.TestCase):
    @staticmethod
    def fake_as_predicted(g, N, val=0):
        nodef, adjacency, edgef = g

        # expand graph size to N
        N = N - nodef.shape[0]
        nodef = np.pad(nodef, [(0,N),(0,0)], 'constant', constant_values=val)
        adjacency = np.copy(adjacency)
        np.fill_diagonal(adjacency, 1)
        adjacency = np.pad(adjacency, [(0,N),(0,N)], 'constant', constant_values=val)
        edgef = np.pad(edgef, [(0,N),(0,N),(0,0)], 'constant', constant_values=val)

        # "invert" softmax and sigmoid in affinity_minitoy (simulating perfect predictions)
        nodef = ecc.one_hot_discretization(nodef,0,1).astype(np.float32) * 3
        adjacency = 10 * (adjacency - 0.5)

        return nodef[np.newaxis,...], adjacency[np.newaxis,...], edgef[np.newaxis,...]

    @staticmethod
    def to_torch(g):
        nodef, adjacency, edgef = g
        return torch.from_numpy(nodef), torch.from_numpy(adjacency), torch.from_numpy(edgef)

    def setUp(self):
        random.seed(1)
        np.random.seed(1)

    def xtest_same_graph(self):
        Args = collections.namedtuple('Args', 'pc_augm_noise, pc_augm_scale, pc_augm_rot, pc_augm_mirror_prob')
        args = Args(pc_augm_noise=0.1, pc_augm_scale=0, pc_augm_rot=0, pc_augm_mirror_prob=0)
        rs = np.random.RandomState(seed=1)
        N = 4

        gP0 = datasets.toy_graph_creator(args, rs, None, clst=0, npts=3, nmarked=2, diameter=1)
        gP = self.fake_as_predicted(gP0, N, 0)

        As = affinity_minitoy(self.to_torch(gP), [self.to_torch(gP0)], True)
        Xs = max_pooling_matching(As)
        print('test_same_graph', Xs, max_pooling_matching(As,discretize=True)) # should be nicely matched

    def xtest_different_graph_same_seed(self):
        Args = collections.namedtuple('Args', 'pc_augm_noise, pc_augm_scale, pc_augm_rot, pc_augm_mirror_prob')
        args = Args(pc_augm_noise=0.1, pc_augm_scale=0, pc_augm_rot=0, pc_augm_mirror_prob=0)
        rs = np.random.RandomState(seed=1)
        N = 4

        gP0 = datasets.toy_graph_creator(args, rs, None, clst=0, npts=3, nmarked=2, diameter=1)
        gP = self.fake_as_predicted(gP0, N, 0)
        rs = np.random.RandomState(seed=1)
        gT = datasets.toy_graph_creator(args, rs, None, clst=0, npts=3, nmarked=2, diameter=1)

        As = affinity_minitoy(self.to_torch(gP), [self.to_torch(gT)], True)
        Xs = max_pooling_matching(As)
        print('test_different_graph_same_seed', Xs, max_pooling_matching(As,discretize=True)) # should be nicely matched


    def xtest_different_graph_diff_seed(self):
        Args = collections.namedtuple('Args', 'pc_augm_noise, pc_augm_scale, pc_augm_rot, pc_augm_mirror_prob')
        args = Args(pc_augm_noise=0.2, pc_augm_scale=1.1, pc_augm_rot=0, pc_augm_mirror_prob=0)
        rs = np.random.RandomState(seed=1)
        N = 3

        gP0 = datasets.toy_graph_creator(args, rs, None, clst=0, npts=3, nmarked=2, diameter=1)
        gP = self.fake_as_predicted(gP0, N, 0)
        gT = datasets.toy_graph_creator(args, rs, None, clst=0, npts=3, nmarked=2, diameter=1)

        As = affinity_minitoy(self.to_torch(gP), [self.to_torch(gT)], False)
        Xs = max_pooling_matching(As)

        print('test_different_graph_diff_seed', Xs, max_pooling_matching(As,discretize=True)) # marked nodes match differently and so new edges are encouraged

        As = affinity_minitoy(self.to_torch(gP), [self.to_torch(gT)], True)
        Xs = max_pooling_matching(As)
        print('test_different_graph_diff_seed', Xs, max_pooling_matching(As,discretize=True)) # marked nodes match differently but edges are nearly locked (and node's xent too strong), so resorts to non-optimal mapping, resolved by hungarian

        Xs = max_pooling_matching(As, discretize=True)
        losses = matching_loss([Variable(t) for t in self.to_torch(gP)], [[Variable(t) for t in self.to_torch(gT)]], [Variable(t) for t in Xs])
        print(losses)

    def xtest_heuristics(self):
        Args = collections.namedtuple('Args', 'pc_augm_noise, pc_augm_scale, pc_augm_rot, pc_augm_mirror_prob')
        args = Args(pc_augm_noise=0, pc_augm_scale=0, pc_augm_rot=0, pc_augm_mirror_prob=0)
        rs = np.random.RandomState(seed=1)
        N = 5
        for clst in range(3):
            npts, nmarked, diameter = 4, 2, 5
            label = np.hstack([ecc.one_hot_discretization(np.array([clst]),0,2), np.array([[0, nmarked, diameter, 0]])]).astype(np.float32)
            gP0 = datasets.toy_graph_creator(args, rs, None, clst=clst, npts=npts, nmarked=nmarked, diameter=diameter)
            gP = self.fake_as_predicted(gP0, N, 0)

            cost = datasets.minitoy_heuristics_cost(label, gP)
            print(clst, cost[0]) # should be nicely matched

if __name__ == "__main__":
    #import pydevd; pydevd.settrace('localhost', port=3333 , stdoutToServer=True, stderrToServer=True, suspend=False)
    unittest.main()
