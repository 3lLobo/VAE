import numpy as np


def mk_sparse_graph_ds(n: int, e: int, d_e: int, batch_size: int=1, batches: int=1):
    """
    Function to create random graph dataset in sparse matrix form.
    We generate the each subject (s), relation (r), object (o) vector separate and then stack and permute them.
    Output shape is [batches*(bs,[s,r,o])].
    Args:
        n: number of nodes.
        e: number of edges between nodes.
        d_e: number of edge attributes.
        batch_size: well, the batch size.
        batches: optional for unicorn dust.
    """
    ds = list()
    for _ in range(batches):
        s = np.random.choice(n, (batch_size, e))
        r = np.random.choice(d_e, (batch_size, e))
        o = np.random.choice(n, (batch_size, e))
        ds.append(np.stack([s,r,o], axis=-1))
    return ds
