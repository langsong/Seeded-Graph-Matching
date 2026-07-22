import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
import networkx as nx
from config import *
from FeatureEngineering.model import predict_seed_score
import joblib

RF_MODEL = None

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

def jaccard_cluster_seeds(G1, G2, n_seeds, optimal_permutation):
    """
    Selects seeds by growing a connected cluster.

    Starts from a random node. At each step, considers the seed set,
    examines all its unselected neighbors, and adds the neighbor with the highest
    Jaccard neighborhood similarity to the seed it is adjacent to.

    If the current seed cluster has no unselected neighbors, a new random
    node is chosen to start another cluster.
    """
    if n_seeds == 0:
        return np.array([]), np.array([])

    n = G1.shape[0]

    # Precompute neighbor sets
    neighbors = [
        set(np.where(G1[i] > 0)[0])
        for i in range(n)
    ]

    # Pick the first seed randomly
    first_seed = np.random.randint(n)

    seeds = [first_seed]
    seed_set = {first_seed}

    while len(seeds) < n_seeds:

        best_candidate = None
        best_score = -1

        # Look at every current seed
        for seed in seeds:

            for candidate in neighbors[seed]:

                if candidate in seed_set:
                    continue

                intersection = len(neighbors[seed] & neighbors[candidate])
                union = len(neighbors[seed] | neighbors[candidate])

                score = intersection / union if union > 0 else 0

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

        # If no neighboring candidates remain, restart from a random node
        if best_candidate is None:

            remaining = list(set(range(n)) - seed_set)

            if not remaining:
                break

            best_candidate = np.random.choice(remaining)

        seeds.append(best_candidate)
        seed_set.add(best_candidate)

    seeds_G1 = np.array(seeds)
    seeds_G2 = optimal_permutation[seeds_G1]

    return seeds_G1, seeds_G2



def triangle_degree_ratio_seeds(G1, G2, n_seeds, optimal_permutation):
    """
    Selects seeds based on the ratio of a node's degree to the number of triangles 
    it participates in plus 1 within G1. Pairs the selected nodes with their correct 
    matches in G2 using the optimal_permutation.
    """
    # Convert G1 from a NumPy adjacency matrix to a NetworkX graph
    G_nx = nx.from_numpy_array(G1)

    # Calculate degrees and number of triangles for all nodes
    degrees = dict(G_nx.degree())
    triangles = nx.triangles(G_nx)
    
    scores = {}
    for node in G_nx.nodes():
        deg = degrees[node]
        tri = triangles[node]
        
        # Calculate score by adding 1 to the denominator
        scores[node] = deg / (tri + 1)

    # Sort nodes by their ratio score in descending order
    sorted_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Pick the top n_seeds nodes from G1
    seeds_g1 = np.array(sorted_nodes[:n_seeds], dtype=int)
    
    # Map them to their correct counterparts in G2 using the optimal permutation
    seeds_g2 = optimal_permutation[seeds_g1]
    
    return seeds_g1, seeds_g2

class RFHillClimber:
    __name__ = "rf_hill_climb_seeds"
    def __init__(self, model):
        """
        Stores the trained Random Forest model.
        """
        self.model = model

    def __call__(
        self,
        G1,
        G2,
        n_seeds,
        optimal_permutation
    ):
        """
        Uses the trained Random Forest to hill climb toward
        a better seed set.
        """

        if n_seeds == 0:
            return np.array([]), np.array([])

        G1_nx = nx.from_numpy_array(G1)
        G2_nx = nx.from_numpy_array(G2)

        # Start from highest-degree seeds
        seeds_G1, seeds_G2 = highest_degree_seeds(
            G1,
            G2,
            n_seeds,
            optimal_permutation,
        )

        current_score = predict_seed_score(
            self.model,
            G1_nx,
            G2_nx,
            seeds_G1,
            seeds_G2
        )

        n_nodes = G1.shape[0]

        for _ in range(100):

            candidate = seeds_G1.copy()

            # Pick one seed to replace
            replace_index = np.random.randint(n_seeds)

            # Pick a node that isn't already a seed
            non_seeds = np.setdiff1d(
                np.arange(n_nodes),
                candidate
            )

            new_node = np.random.choice(non_seeds)

            candidate[replace_index] = new_node

            candidate_score = predict_seed_score(
                self.model,
                G1_nx,
                G2_nx,
                candidate,
                optimal_permutation[candidate],
            )

            if candidate_score > current_score:
                seeds_G1 = candidate
                current_score = candidate_score

        seeds_G2 = optimal_permutation[seeds_G1]

        return seeds_G1, seeds_G2

def get_rf_model():
    """
    Lazily loads the trained Random Forest model.
    Each worker process loads it only once.
    """
    global RF_MODEL

    if RF_MODEL is None:
        RF_MODEL = joblib.load("rf_seed_model.pkl")

    return RF_MODEL

def rf_random_seeds(
    G1,
    G2,
    n_seeds,
    optimal_permutation,
    attempts=10,
):
    """
    Generates several random seed sets and uses the trained
    Random Forest model to choose the one predicted to
    produce the best graph matching accuracy.
    """

    if n_seeds == 0:
        return np.array([]), np.array([])

    model = get_rf_model()

    G1_nx = nx.from_numpy_array(G1)
    G2_nx = nx.from_numpy_array(G2)

    best_score = -np.inf
    best_seeds_G1 = None
    best_seeds_G2 = None

    for _ in range(attempts):

        # Generate a random seed set
        seeds_G1, seeds_G2 = random_seeds(
            G1,
            G2,
            n_seeds,
            optimal_permutation,
        )

        # Predict matching performance
        score = predict_seed_score(
            model,
            G1_nx,
            G2_nx,
            seeds_G1,
            seeds_G2,
        )

        # Keep the best predicted seed set
        if score > best_score:
            best_score = score
            best_seeds_G1 = seeds_G1
            best_seeds_G2 = seeds_G2

    return best_seeds_G1, best_seeds_G2

def rf_greedy_seeds(
    G1,
    G2,
    n_seeds,
    optimal_permutation,
    candidates_per_step=50,
):
    """
    Iteratively builds a seed set by using the Random Forest
    model to select the most promising next seed.

    At each step:
        1. Generate candidate seed pairs.
        2. Temporarily add each candidate.
        3. Predict final matching performance with RF.
        4. Keep the candidate with the highest predicted score.

    Parameters
    ----------
    candidates_per_step : int
        Number of random candidate seed pairs to evaluate
        at each greedy step.
    """

    if n_seeds == 0:
        return np.array([]), np.array([])

    model = get_rf_model()

    G1_nx = nx.from_numpy_array(G1)
    G2_nx = nx.from_numpy_array(G2)

    # Store chosen seeds
    selected_G1 = []
    selected_G2 = []

    # Track vertices already selected
    available_G1 = set(range(len(G1)))
    available_G2 = set(range(len(G2)))

    for _ in range(n_seeds):

        best_score = -np.inf
        best_seed_G1 = None
        best_seed_G2 = None

        # Generate candidate seed pairs
        candidate_pairs = []

        for _ in range(candidates_per_step):

            node_G1 = np.random.choice(
                list(available_G1)
            )

            # Use optimal permutation to get the true match
            node_G2 = optimal_permutation[node_G1]

            if node_G2 in available_G2:
                candidate_pairs.append(
                    (node_G1, node_G2)
                )

        # Evaluate candidates
        for candidate_G1, candidate_G2 in candidate_pairs:

            test_seeds_G1 = np.append(
                selected_G1,
                candidate_G1
            )

            test_seeds_G2 = np.append(
                selected_G2,
                candidate_G2
            )

            score = predict_seed_score(
                model,
                G1_nx,
                G2_nx,
                test_seeds_G1,
                test_seeds_G2,
            )

            if score > best_score:
                best_score = score
                best_seed_G1 = candidate_G1
                best_seed_G2 = candidate_G2

        # Add the best candidate
        selected_G1.append(best_seed_G1)
        selected_G2.append(best_seed_G2)

        available_G1.remove(best_seed_G1)
        available_G2.remove(best_seed_G2)

    return (
        np.array(selected_G1),
        np.array(selected_G2),
    )