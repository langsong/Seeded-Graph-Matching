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
    shuffle_perm = np.random.permutation(n)
    
    # Apply the permutation to both rows and columns of G2
    G2_shuffled = G2[shuffle_perm, :][:, shuffle_perm]

    # Compute unshuffling
    optimal_permutation = np.argsort(shuffle_perm)
    
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


def gen_IER_graphs(
    n=N_ER_NODES,
    rho=ER_RHO,
    p_min=0.0,
    p_max=1.0,
    directed=False,
    loops=False,
):
    """Generate and shuffle a correlated inhomogeneous Erdős-Rényi pair.

    Each edge probability is sampled independently from
    ``Uniform(p_min, p_max)``. For an undirected graph, one probability is
    sampled for each unordered vertex pair and mirrored across the diagonal.

    Conditional on the resulting probability matrix ``P``, graph A has
    independent Bernoulli(P[i, j]) edges. Graph B is sampled so that it has
    the same marginal probabilities and edge-wise correlation ``rho`` with A.
    Graph B is then randomly relabelled, and the returned permutation maps
    vertices in graph A to their observed labels in the shuffled graph B.

    The default interval gives the dense Uniform(0, 1) model. For a sparse
    model with desired mean edge probability ``p_bar <= 0.5``, use
    ``p_min=0`` and ``p_max=2 * p_bar``.
    """

    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(rho, (int, float, np.integer, np.floating)):
        raise TypeError("rho must be numeric.")
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must be between 0 and 1.")
    for name, value in (("p_min", p_min), ("p_max", p_max)):
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be numeric.")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if p_min > p_max:
        raise ValueError("p_min must be less than or equal to p_max.")
    if not isinstance(directed, (bool, np.bool_)):
        raise TypeError("directed must be boolean.")
    if not isinstance(loops, (bool, np.bool_)):
        raise TypeError("loops must be boolean.")

    probability_matrix = np.zeros((n, n), dtype=float)
    if directed:
        probability_matrix = np.random.uniform(p_min, p_max, size=(n, n))
        if not loops:
            np.fill_diagonal(probability_matrix, 0.0)
    else:
        upper_rows, upper_columns = np.triu_indices(n, k=1)
        upper_probabilities = np.random.uniform(
            p_min, p_max, size=len(upper_rows)
        )
        probability_matrix[upper_rows, upper_columns] = upper_probabilities
        probability_matrix[upper_columns, upper_rows] = upper_probabilities
        if loops:
            diagonal = np.arange(n)
            probability_matrix[diagonal, diagonal] = np.random.uniform(
                p_min, p_max, size=n
            )

    correlation_matrix = np.full((n, n), float(rho), dtype=float)
    graph_1, graph_2 = sample_edges_corr(
        probability_matrix,
        correlation_matrix,
        directed=directed,
        loops=loops,
    )

    shuffle_permutation = np.random.permutation(n)
    graph_2_shuffled = graph_2[shuffle_permutation][:, shuffle_permutation]
    optimal_permutation = np.argsort(shuffle_permutation)

    return graph_1, graph_2_shuffled, optimal_permutation


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
        expected_degrees = expected_degrees * (sum_deg / max_deg ** 2) * 0.95
    
    # Extract the probability matrix from the expected degree layout
    # For a given node pair, probability = (d_i * d_j) / sum(d)
    deg_matrix = np.outer(expected_degrees, expected_degrees) / np.sum(expected_degrees)
    np.clip(deg_matrix, 0, 1, out=deg_matrix) # Clip boundaries just in case

    # Graph correlation in matrix form
    corr_mat = np.full((n, n), rho)

    # 4. Use graspologic to sample a pair of correlated graphs from the probability matrix
    G1, G2 = sample_edges_corr(deg_matrix, corr_mat, directed=directed, loops=loops)
    
    # 5. Shuffle G2 to simulate an actual graph matching problem
    shuffle_perm = np.random.permutation(n)
    G2_shuffled = G2[shuffle_perm][:, shuffle_perm]

    # Compute unshuffling
    optimal_permutation = np.argsort(shuffle_perm)
    
    return G1, G2_shuffled, optimal_permutation

def gen_PAPER_graphs(n=N_PPR_NODES, alpha=PPR_ALPHA, p=PPR_EDGE_PROBABILITY, rho=PPR_RHO, directed=False, loops=False):
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
        expected_degrees = expected_degrees * (sum_deg / max_deg ** 2) * 0.95
    
    # Extract the probability matrix from the expected degree layout
    # For a given node pair, probability = (d_i * d_j) / sum(d)
    deg_matrix = np.outer(expected_degrees, expected_degrees) / np.sum(expected_degrees)

    # Add in Erdős-Rényi style edge probability
    deg_matrix = deg_matrix + p
    
    np.clip(deg_matrix, 0, 1, out=deg_matrix) # Clip boundaries

    # Graph correlation in matrix form
    corr_mat = np.full((n, n), rho)

    # 4. Use graspologic to sample a pair of correlated graphs from the probability matrix
    G1, G2 = sample_edges_corr(deg_matrix, corr_mat, directed=directed, loops=loops)
    
    # 5. Shuffle G2 to simulate an actual graph matching problem
    shuffle_perm = np.random.permutation(n)
    G2_shuffled = G2[shuffle_perm][:, shuffle_perm]

    # Compute unshuffling
    optimal_permutation = np.argsort(shuffle_perm)
    
    return G1, G2_shuffled, optimal_permutation
