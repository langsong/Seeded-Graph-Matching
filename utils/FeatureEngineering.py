import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestRegressor







def compute_seed_features(G1, G2, seeds_G1, seeds_G2):
    """
    Computes features describing a pair of seed sets across two graphs.

    Parameters
    ----------
    G1, G2 : networkx.Graph
        The two graphs being matched.

    seeds_G1, seeds_G2 : iterable
        Corresponding seed sets in G1 and G2.

    Returns
    -------
    dict
        Feature dictionary for one graph matching trial. Looks like
        {
            G1_avg_degree: _
            G2_avg_degree: _
            ...
        }
    """

    features = {}

    features_G1 = compute_single_graph_seed_features(G1, seeds_G1)
    features_G2 = compute_single_graph_seed_features(G2, seeds_G2)

    for key in features_G1:

        value_G1 = features_G1[key]
        value_G2 = features_G2[key]

        features[f"G1_{key}"] = value_G1
        features[f"G2_{key}"] = value_G2
        features[f"{key}_difference"] = abs(value_G1 - value_G2)

    return features


def compute_single_graph_seed_features(G, seeds):
    """
    Computes structural properties of a seed set inside one graph.
    """

    features = {}

    features.update(compute_degree_features(G, seeds))
    features.update(compute_neighbor_features(G, seeds))
    features.update(compute_clustering_features(G, seeds))
    features.update(compute_distance_features(G, seeds))
    features.update(compute_exposure_features(G, seeds))

    return features


def compute_degree_features(G, seeds):
    """
    Degree statistics of seed nodes.
    """

    degrees = np.array([
        G.degree(node)
        for node in seeds
    ])

    return {
        "avg_degree": degrees.mean(),
        "max_degree": degrees.max(),
        "min_degree": degrees.min(),
        "degree_std": degrees.std(),
    }


def compute_neighbor_features(G, seeds):
    """
    Features describing seed neighborhoods.
    """

    seed_set = set(seeds)

    neighbor_degrees = []
    two_hop_sizes = []

    for seed in seeds:

        neighbors = list(G.neighbors(seed))

        if neighbors:
            neighbor_degrees.append(
                np.mean([
                    G.degree(node)
                    for node in neighbors
                ])
            )

        two_hop = set()

        for neighbor in neighbors:
            two_hop.update(G.neighbors(neighbor))

        two_hop -= seed_set

        two_hop_sizes.append(len(two_hop))


    return {
        "avg_neighbor_degree": (
            np.mean(neighbor_degrees)
            if neighbor_degrees else 0
        ),

        "avg_two_hop_size": (
            np.mean(two_hop_sizes)
            if two_hop_sizes else 0
        ),
    }


def compute_clustering_features(G, seeds):
    """
    Local clustering coefficient statistics.
    """

    clustering = np.array([
        nx.clustering(G, node)
        for node in seeds
    ])

    return {
        "avg_clustering": clustering.mean(),
        "clustering_std": clustering.std(),
    }


def compute_distance_features(G, seeds):
    """
    Pairwise shortest path statistics between seeds.
    """

    distances = []

    seeds = list(seeds)

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):

            try:
                distance = nx.shortest_path_length(
                    G,
                    seeds[i],
                    seeds[j]
                )

                distances.append(distance)

            except nx.NetworkXNoPath:
                continue


    if len(distances) == 0:
        return {
            "avg_pairwise_distance": 0,
            "max_pairwise_distance": 0,
            "min_pairwise_distance": 0,
        }


    distances = np.array(distances)

    return {
        "avg_pairwise_distance": distances.mean(),
        "max_pairwise_distance": distances.max(),
        "min_pairwise_distance": distances.min(),
    }


def compute_exposure_features(G, seeds):
    """
    Measures how much of the graph is immediately exposed
    by the seed set.
    """

    seed_set = set(seeds)

    exposed_nodes = set()

    for seed in seeds:
        exposed_nodes.update(G.neighbors(seed))

    exposed_nodes -= seed_set

    return {
        "seed_exposure": (
            len(exposed_nodes) /
            G.number_of_nodes()
        )
    }