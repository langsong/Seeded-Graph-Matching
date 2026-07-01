from graspologic.simulations import sbm_corr, sample_edges, er_corr
import numpy as np
from config import *

def gen_ER_graphs(n=N_ER_NODES, p=EDGE_PROBABILITY, rho=ER_RHO, directed=False, loops=False):
    """
    Generates a pair of correlated Erdős-Rényi graphs (G1, G2).
    """
    G1, G2 = er_corr(
        n=n,
        p=p,
        r=rho,
        directed=directed,
        loops=loops,
    )

    # Randomly permute graph 2
    permutation = np.random.permutation(n)
    G2_shuffled = G2[np.ix_(permutation, permutation)]

    return G1, G2_shuffled, permutation

def gen_SBM_graphs(directed=False, loops=False, n_per_block=N_PER_BLOCK, n_blocks=N_BLOCKS, rho=SBM_RHO, block_probs=BLOCK_PROBS):
    """
    Generates a pair of correlated SBM graphs and shuffles the second graph.
    """
    
    n = [n_per_block] * n_blocks
    
    # Generate correlated SBMs (G1 and G2 are natively perfectly aligned)
    G1, G2 = sbm_corr(n, block_probs, rho, directed=directed, loops=loops)
    
    # Shuffle G2 to simulate an unknown permutation
    total_nodes = sum(n)
    shuffle_perm = np.random.permutation(total_nodes)
    
    # To recover the alignment, we track the inverse of our shuffle (the unshuffle)
    unshuffle_perm = np.argsort(shuffle_perm)
    G2_shuffled = G2[shuffle_perm][:, shuffle_perm]
    
    # The optimal permutation maps node i in G1 to unshuffle_perm[i] in G2_shuffled
    optimal_permutation = unshuffle_perm
    
    return G1, G2_shuffled, optimal_permutation
