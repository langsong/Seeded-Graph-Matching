import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
import networkx as nx
from config import *

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


def spectral_unique_seeds(G1, G2, n_seeds, optimal_permutation):

    embedding = SpectralEmbedding(
        n_components=10,
        affinity="precomputed"
    )

    X = embedding.fit_transform(G1)

    distances = pairwise_distances(X)

    np.fill_diagonal(distances, np.inf)

    uniqueness = np.min(distances, axis=1)

    seeds_G1 = np.argsort(uniqueness)[-n_seeds:]

    seeds_G2 = optimal_permutation[seeds_G1]

    return seeds_G1, seeds_G2


# sum of degrees of all neighbors 
def neighbor_degree_seeds(G1, G2, n_seeds, optimal_permutation):

    degrees = np.sum(G1, axis=1)

    scores = G1 @ degrees

    seeds_G1 = np.argsort(scores)[-n_seeds:]

    seeds_G2 = optimal_permutation[seeds_G1]

    return seeds_G1, seeds_G2

def jaccard_neighborhood_seeds(G1, G2, n_seeds, optimal_permutation):
    """
    Selects the 'n_seeds' vertices from G1 with the highest average
    Jaccard neighborhood similarity among their neighbors and returns
    them along with their true pairs in G2.
    """
    if n_seeds == 0:
        return np.array([]), np.array([])

    n_nodes = G1.shape[0]
    jaccard_scores = np.zeros(n_nodes)

    # Convert adjacency matrix to neighbor sets for faster computation
    neighbors = [
        set(np.where(G1[i] > 0)[0])
        for i in range(n_nodes)
    ]

    # Calculate average neighbor-neighbor Jaccard score for each node
    for node in range(n_nodes):
        node_neighbors = neighbors[node]

        # Nodes with no neighbors cannot have a Jaccard score
        if len(node_neighbors) == 0:
            jaccard_scores[node] = 0
            continue

        total_similarity = 0

        for neighbor in node_neighbors:
            neighbor_neighbors = neighbors[neighbor]

            intersection = len(node_neighbors.intersection(neighbor_neighbors))
            union = len(node_neighbors.union(neighbor_neighbors))

            if union > 0:
                total_similarity += intersection / union

        jaccard_scores[node] = total_similarity / len(node_neighbors)

    # Select nodes with highest Jaccard neighborhood scores
    highest_jaccard_nodes = np.argsort(jaccard_scores)[-n_seeds:][::-1]

    # Map G1 seeds to their true G2 matches
    seeds_G1 = highest_jaccard_nodes
    seeds_G2 = optimal_permutation[seeds_G1]

    return seeds_G1, seeds_G2