from graspologic.simulations import sbm_corr, sample_edges, er_corr
import numpy as np
from graspologic.simulations import sample_edges_corr
from config import *

def gen_ER_graphs(n=N_ER_NODES, p=EDGE_PROBABILITY, rho=ER_RHO, directed=False, loops=False):
    """
    Generates a pair of correlated Erdős-Rényi graphs (G1, G2).
    """
    # 1. Create the base probability matrix for an Erdős-Rényi graph.
    # Every possible edge shares the exact same probability 'p'.
    #P = np.full((n, n), p)
    
    # 2. Sample two correlated child graphs from the base probability matrix
    G1, G2 = er_corr(n, p, rho, directed=directed, loops=loops)
    
    # 3. Create a random permutation to shuffle Graph 2
    optimal_permutation = np.random.permutation(n)
    
    # Apply the permutation to both rows and columns of G2
    G2_shuffled = G2[optimal_permutation, :][:, optimal_permutation]
    
    return G1, G2_shuffled, optimal_permutation

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

def gen_correlated_powerlaw_graphs(n=N_PL_NODES, alpha=ALPHA, rho=PL_RHO, directed=False, loops=False):
    """
    Generates a pair of correlated power-law graphs with a specified correlation rho.
    """
    # 1. Generate a power-law sequence for expected node degrees
    # Add 2 to avoid 0-degree nodes which can mess up matching metrics
    expected_degrees = np.random.pareto(alpha, size=n) + 2
    
    # 2. Scale expected degrees so they form valid probabilities when multiplied
    # P(edge i-j) = (d_i * d_j) / sum(d)
    sum_deg = np.sum(expected_degrees)
    max_deg = np.max(expected_degrees)
    
    # Check to ensure the max probability won't exceed 1.0
    if (max_deg ** 2) / sum_deg > 1.0:
        expected_degrees = expected_degrees * (np.sqrt(sum_deg) / max_deg) * 0.95

    # 3. Create the underlying "parent" probability matrix using NetworkX's generator
    # networkx.expected_degree_graph natively outputs a graph based on the Chung-Lu model
    # nx_parent_graph = nx.expected_degree_graph(expected_degrees.astype(int), selfloops=loops)
    
    # Extract the probability matrix from the expected degree layout
    # For a given node pair, probability = (d_i * d_j) / sum(d)
    deg_matrix = np.outer(expected_degrees, expected_degrees) / np.sum(expected_degrees)
    np.clip(deg_matrix, 0, 1, out=deg_matrix) # Clip boundaries just in case

    # Graph correlation in matrix form
    corr_mat = np.full((n, n), rho)

    # 4. Use graspologic to sample a pair of correlated graphs from the probability matrix
    G1, G2 = sample_edges_corr(deg_matrix, corr_mat, directed=directed, loops=loops)
    
    # 5. Shuffle G2 to simulate an actual graph matching problem
    optimal_perm = np.random.permutation(n)
    G2_shuffled = G2[optimal_perm][:, optimal_perm]
    
    return G1, G2_shuffled, optimal_perm