import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from graspologic.simulations import sbm_corr
from graspologic.match import graph_match

def gen_SBM_graphs(directed=False, loops=False, n_per_block=150, n_blocks=3, rho=0.9, block_probs=None):
    """
    Generates a pair of correlated SBM graphs and shuffles the second graph.
    """
    if block_probs is None:
        # Default probabilities as outlined in the SGM tutorial
        block_probs = np.array([
            [0.7, 0.3, 0.4],
            [0.3, 0.7, 0.3],
            [0.4, 0.3, 0.7]
        ])
    
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

def blocked_random_seeds(G1, G2, n_seeds, optimal_permutation, n_blocks=3):
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

def match_ratio(predicted_permutation, optimal_permutation):
    """
    Computes the fraction of vertices correctly matched.
    """
    return np.mean(predicted_permutation == optimal_permutation)

def compare_seeding(graph_gen_func, seeding_funcs_list, seed_nums_list, n_trials=10):
    """
    Runs graph matching trials, extracts confidence intervals, and plots the results.
    """
    # Initialize a dictionary to hold all results
    results = {func.__name__: {s: [] for s in seed_nums_list} for func in seeding_funcs_list}
    
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
                _, perm_inds, _, _ = graph_match(G1, G2_shuffled, partial_match=partial_match)
                
                # 4. Compute and Store Score
                score = match_ratio(perm_inds, optimal_perm)
                results[seeding_func.__name__][s].append(score)
                
    # --- Plotting the Results ---
    plt.figure(figsize=(10, 6))
    
    for func_name in results:
        means = []
        ci_lower = []
        ci_upper = []
        
        for s in seed_nums_list:
            data = results[func_name][s]
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
    plt.title('Seeded Graph Matching: Random vs. Blocked Random Seeding', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # Settings mapped from the tutorial context provided
    seed_counts_to_test = [0, 2, 4, 6, 8]
    trials_per_seed_count = 50  # Reduced to 10 for timely execution, increase as desired
    
    # print("Initiating SGM Experiments... this might take a minute depending on core count.")
    compare_seeding(
        graph_gen_func=gen_SBM_graphs,
        seeding_funcs_list=[random_seeds, blocked_random_seeds],
        seed_nums_list=seed_counts_to_test,
        n_trials=trials_per_seed_count
    )