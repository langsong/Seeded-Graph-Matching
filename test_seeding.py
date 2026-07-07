import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as st
from graspologic.simulations import sbm_corr
from graspologic.simulations import er_corr
from graspologic.simulations import sample_edges_corr
from graspologic.match import graph_match
import networkx as nx

# --- Constants ---

np.random.seed(1234)

# SBM Constants
N_PER_BLOCK = 200
N_BLOCKS = 3
SBM_RHO = .5
BLOCK_PROBS = np.array([
            [0.7, 0.3, 0.4],
            [0.3, 0.7, 0.3],
            [0.4, 0.3, 0.7]
        ])

#Erdos Renyi Constants
N_ER_NODES = 600
EDGE_PROBABILITY = .2
ER_RHO = .8

#Power Law Constants
N_PL_NODES = 600
ALPHA = 2.5
PL_RHO = .8

#Experiment Constants
TRIALS_PER_SEED_NUMBER = 25
SEED_COUNTS = [40, 60, 100, 150, 250, 400]

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

def gen_ER_graphs(n=N_ER_NODES, p=EDGE_PROBABILITY, rho=ER_RHO, directed=False, loops=False):
    """
    Generates a pair of correlated Erdős-Rényi graphs (G1, G2).
    """
    
    # 1. Sample two correlated ER Graphs
    G1, G2 = er_corr(n, p, rho, directed=directed, loops=loops)
    
    # 2. Create a random permutation to shuffle Graph 2
    optimal_permutation = np.random.permutation(n)
    
    # 3. Apply the permutation to both rows and columns of G2
    G2_shuffled = G2[optimal_permutation][:, optimal_permutation]
    
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

def random_seeds(G1, G2, n_seeds, optimal_permutation):
    """
    Randomly selects 'n_seeds' vertices from G1 and their true pairs in G2.
    """
    if n_seeds == 0:
        return np.array([]), np.array([])
        
    n_nodes = G1.shape[0]
    seeds_G1 = np.random.choice(n_nodes, n_seeds, replace=False)
    seeds_G2 = optimal_permutation[seeds_G1]
    
    return seeds_G1, seeds_G2

def highest_degree_seeds(G1, G2, n_seeds, optimal_permutation):
    """
    Selects the 'n_seeds' vertices from G1 with the highest total degrees 
    and returns them along with their true pairs in G2.
    """
    if n_seeds == 0:
        return np.array([]), np.array([])
        
    # Calculate the degree of each node in G1. 
    degrees = np.sum(G1, axis=1)
    
    # Get the indices that would sort the degrees in ascending order, 
    # then take the last 'n_seeds' elements and reverse them to get highest-to-lowest.
    highest_degree_nodes = np.argsort(degrees)[-n_seeds:][::-1]
    
    # Map the highest degree G1 vertices to their corresponding shuffled vertices in G2
    seeds_G1 = highest_degree_nodes
    seeds_G2 = optimal_permutation[seeds_G1]
    
    return seeds_G1, seeds_G2

def blocked_random_seeds(G1, G2, n_seeds, optimal_permutation, n_blocks=N_BLOCKS):
    """
    Selects seeds distributed evenly across the SBM blocks.
    """
    if n_seeds == 0:
        return np.array([]), np.array([])
        
    n_nodes = G1.shape[0]
    nodes_per_block = n_nodes // n_blocks
    seeds_per_block = n_seeds // n_blocks
    
    seeds_G1 = []
    
    # Pull an equal number of seeds from each block
    for b in range(n_blocks):
        block_start = b * nodes_per_block
        block_end = (b + 1) * nodes_per_block
        block_nodes = np.arange(block_start, block_end)
        
        b_seeds = np.random.choice(block_nodes, seeds_per_block, replace=False)
        seeds_G1.extend(b_seeds)
        
    # Handle remainder if n_seeds is not perfectly divisible by n_blocks
    remainder = n_seeds - (seeds_per_block * n_blocks)
    if remainder > 0:
        remaining_nodes = np.setdiff1d(np.arange(n_nodes), seeds_G1)
        extra_seeds = np.random.choice(remaining_nodes, remainder, replace=False)
        seeds_G1.extend(extra_seeds)
        
    seeds_G1 = np.array(seeds_G1)
    seeds_G2 = optimal_permutation[seeds_G1]
    
    return seeds_G1, seeds_G2

def blocked_highest_degree_seeds(G1, G2, n_seeds, optimal_permutation, n_blocks=N_BLOCKS, n_per_block=N_PER_BLOCK):
    """
    Selects an equal number of seeds from each SBM block, prioritizing the 
    highest total degree nodes within each block.
    """
    if n_seeds == 0:
        return np.array([]), np.array([])
        
    # Calculate how many seeds must be pulled from each block
    seeds_per_block = n_seeds // n_blocks
    remainder = n_seeds % n_blocks
        
    # Calculate the degree of every node in G1
    degrees = np.sum(G1, axis=1)
    
    seeds_G1 = []
    
    # Iterate through each block using the equal block sizes
    for b in range(n_blocks):
        current_start_idx = b * n_per_block
        current_end_idx = current_start_idx + n_per_block
        
        # Isolate the degrees corresponding only to the current block's nodes
        block_degrees = degrees[current_start_idx:current_end_idx]

        # Calculate number of seeds in current block
        block_seeds = seeds_per_block + 1 if b < remainder else seeds_per_block
        
        # Sort indices within the block by degree in ascending order,
        # extract the top required number of elements, and reverse for highest-to-lowest
        top_block_indices = np.argsort(block_degrees)[-block_seeds:][::-1]
        
        # Map the block-relative indices back to absolute G1 node indices
        absolute_indices = top_block_indices + current_start_idx
        seeds_G1.extend(absolute_indices)
        
    # If the divisions didn't match perfectly (e.g. fallback gave too many), 
    # slice down to the exact requested n_seeds
    seeds_G1 = np.array(seeds_G1)[:n_seeds]
    
    # Map the chosen G1 seed vertices to their corresponding shuffled vertices in G2
    seeds_G2 = optimal_permutation[seeds_G1]
    
    return seeds_G1, seeds_G2

def betweenness_seeds(G1, G2, n_seeds, optimal_permutation):
    """
    Selects seeds based on the highest betweenness centrality in Graph 1
    """
    # 1. Convert G1 from a NumPy adjacency matrix to a NetworkX Graph
    G1_nx = nx.from_numpy_array(G1)
    
    # 2. Calculate the betweenness centrality of each node (uses Brandes' Algorithm)
    betweenness_dict = nx.betweenness_centrality(G1_nx)
    
    # 3. Sort node indices by their betweenness centrality in descending order
    sorted_nodes = sorted(betweenness_dict.keys(), key=lambda node: betweenness_dict[node], reverse=True)
    
    # 4. Extract the top n_seeds vertices with the highest betweenness
    seeds_g1 = sorted_nodes[:n_seeds]
    
    # 5. Retrieve their true counterpart pairings for Graph 2 using the unshuffling array
    seeds_g2 = [optimal_permutation[v] for v in seeds_g1]
    
    # Return as two aligned numpy arrays (or np.column_stack if your script expects a 2D array)
    return np.array(seeds_g1), np.array(seeds_g2)

def match_ratio(predicted_permutation, optimal_permutation):
    """
    Computes the fraction of vertices correctly matched.
    """
    return np.mean(predicted_permutation == optimal_permutation)

def compare_seeding(graph_gen_func, seeding_funcs_list, seed_nums_list, n_trials=TRIALS_PER_SEED_NUMBER):
    """
    Runs graph matching trials, extracts confidence intervals, and plots the results.
    """
    # Initialize a dictionary to hold all results
    results_accuracy = {func.__name__: {num: [] for num in seed_nums_list} for func in seeding_funcs_list}
    results_runtime = {func.__name__: {num: [] for num in seed_nums_list} for func in seeding_funcs_list}
    
    for s in seed_nums_list:
        print(f"Running trials for {s} seeds...")
        
        for trial in range(n_trials):
            # 1. Generate Graphs
            G1, G2_shuffled, optimal_perm = graph_gen_func()
            
            for seeding_func in seeding_funcs_list:
                # 2. Grab Seeds
                seeds_G1, seeds_G2 = seeding_func(G1, G2_shuffled, s, optimal_perm)
                partial_match = np.column_stack((seeds_G1, seeds_G2))
                
                # 3. Fit Graph Match Model
                start_time = time.process_time()
                _, perm_inds, _, _ = graph_match(G1, G2_shuffled, partial_match=partial_match)
                end_time = time.process_time()

                # 4. Calculate elapsed time
                elapsed_time = end_time - start_time
                results_runtime[seeding_func.__name__][s].append(elapsed_time)
                
                # 4. Compute and Store Score
                score = match_ratio(perm_inds, optimal_perm)
                results_accuracy[seeding_func.__name__][s].append(score)
                
    # --- Plotting the Accuracy ---
    plt.figure(figsize=(10, 6))
    
    for func_name in results_accuracy:
        means = []
        ci_lower = []
        ci_upper = []
        
        for s in seed_nums_list:
            data = results_accuracy[func_name][s]
            mean_val = np.mean(data)
            
            # Calculate 95% Confidence Interval using Standard Error
            se = st.sem(data) if len(data) > 1 else 0
            ci = se * 1.96 
            
            means.append(mean_val)
            ci_lower.append(mean_val - ci)
            ci_upper.append(mean_val + ci)
            
        # Plot mean lines
        plt.plot(seed_nums_list, means, label=func_name, marker='o')
        # Plot confidence intervals
        plt.fill_between(seed_nums_list, ci_lower, ci_upper, alpha=0.2)
        
    plt.xlabel('Number of Seeds', fontsize=12)
    plt.ylabel('Match Ratio (Accuracy)', fontsize=12)
    plt.title('Match Ratio by Seeding Strategy', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- Plotting the Runtime ---
    plt.figure(figsize=(10, 6))
    
    for func_name in results_runtime:
        means = []
        ci_lower = []
        ci_upper = []
        
        for s in seed_nums_list:
            data = results_runtime[func_name][s]
            mean_val = np.mean(data)
            
            # Calculate 95% Confidence Interval using Standard Error
            se = st.sem(data) if len(data) > 1 else 0
            ci = se * 1.96 
            
            means.append(mean_val)
            ci_lower.append(mean_val - ci)
            ci_upper.append(mean_val + ci)
            
        # Plot mean lines
        plt.plot(seed_nums_list, means, label=func_name, marker='o')
        # Plot confidence intervals
        plt.fill_between(seed_nums_list, ci_lower, ci_upper, alpha=0.2)
        
    plt.xlabel('Number of Seeds', fontsize=12)
    plt.ylabel('Runtime', fontsize=12)
    plt.title('Runtime by seeding strategy', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return results_accuracy, results_runtime

if __name__ == "__main__":    
    # print("Initiating SGM Experiments... this might take a minute depending on core count.")
    compare_seeding(
        graph_gen_func=gen_correlated_powerlaw_graphs,
        seeding_funcs_list=[random_seeds, highest_degree_seeds, betweenness_seeds],
        seed_nums_list=SEED_COUNTS,
        n_trials=TRIALS_PER_SEED_NUMBER
    )