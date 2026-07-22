import numpy as np
import networkx as nx
def seed_neighborhood_coverage(G1, G2, seeds_G1, seeds_G2, optimal_permutation):
    """
    Computes the neighborhood coverage of a seed set.

    The coverage is the number of unique vertices adjacent to at least
    one seed (excluding the seeds themselves).

    Parameters
    ----------
    G1 : ndarray
        Adjacency matrix of Graph 1.

    seeds_G1 : array-like
        Indices of the seed vertices.

    Returns
    -------
    int
        Number of unique non-seed vertices adjacent to at least one seed.
    """
    n = G1.shape[0]
    possible = n - len(seeds_G1)
    seeds = set(seeds_G1)
    covered = set()

    for seed in seeds:
        neighbors = np.where(G1[seed] > 0)[0]

        for neighbor in neighbors:
            if neighbor not in seeds:
                covered.add(neighbor)

    return len(covered) / possible if possible > 0 else 0


def multi_seed_exposure(G1, G2, seeds_G1, seeds_G2, optimal_permutation, threshold=2):
    """
    Computes how many non-seed vertices have at least `r`
    neighbors that are part of the seed set.

    This measures how much initial percolation support the seed set provides.

    Parameters
    ----------
    G1 : ndarray
        Adjacency matrix of Graph 1.

    seeds_G1 : array-like
        Indices of seed vertices.

    threshold : int
        Number of seed neighbors required.

    Returns
    -------
    int
        Number of non-seed vertices with at least `threshold`
        seed neighbors.
    """

    seeds = set(seeds_G1)

    exposure_count = 0

    for node in range(G1.shape[0]):

        # Ignore seed nodes themselves
        if node in seeds:
            continue

        # Count how many neighbors are seeds
        neighbors = np.where(G1[node] > 0)[0]

        seed_neighbors = sum(
            neighbor in seeds
            for neighbor in neighbors
        )

        if seed_neighbors >= threshold:
            exposure_count += 1

    return exposure_count

def initial_mark_potential(G1, G2, seeds_G1, seeds_G2, optimal_permutation, threshold=2):
    """
    Computes the number of correctly matched non-seed vertices that already
    have enough seed support to satisfy the percolation threshold.

    Parameters
    ----------
    G1 : ndarray
        Adjacency matrix of Graph 1.

    G2 : ndarray
        Adjacency matrix of Graph 2.

    seeds_G1 : array-like
        Seed vertices in G1.

    seeds_G2 : array-like
        Corresponding seed vertices in G2.

    optimal_permutation : ndarray
        True mapping from G1 vertices to G2 vertices.

    threshold : int
        Required number of supporting seed matches.

    Returns
    -------
    int
        Number of non-seed vertices whose true match has at least
        `threshold` supporting seed neighbors.
    """

    seed_set = set(seeds_G1)

    # Map G1 seed -> G2 seed
    seed_mapping = dict(zip(seeds_G1, seeds_G2))

    potential = 0

    for u in range(G1.shape[0]):

        # Skip existing seeds
        if u in seed_set:
            continue

        v = optimal_permutation[u]

        # Neighbors of u in G1
        neighbors_u = np.where(G1[u] > 0)[0]

        support = 0

        for neighbor in neighbors_u:

            # Only count already matched neighbors
            if neighbor in seed_mapping:

                mapped_neighbor = seed_mapping[neighbor]

                # Check if the mapped seed neighbor is also
                # connected to v in G2
                if G2[v, mapped_neighbor] > 0:
                    support += 1

        if support >= threshold:
            potential += 1

    return potential

def seed_dispersion(G1, G2, seeds_G1, seeds_G2, optimal_permutation):
    """
    Computes seed dispersion as the average shortest path distance
    between all pairs of seeds.

    Parameters
    ----------
    G1 : ndarray
        Adjacency matrix of Graph 1.

    seeds_G1 : array-like
        Indices of seed vertices.

    Returns
    -------
    float
        Average shortest path distance between seed pairs.
    """

    # Convert adjacency matrix to NetworkX graph
    G = nx.from_numpy_array(G1)

    seeds = list(seeds_G1)

    # Need at least two seeds
    if len(seeds) < 2:
        return 0

    distances = []

    # Compute shortest path between every pair of seeds
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):

            try:
                distance = nx.shortest_path_length(
                    G,
                    source=seeds[i],
                    target=seeds[j]
                )

                distances.append(distance)

            except nx.NetworkXNoPath:
                # If disconnected, treat as maximally far
                distances.append(G.number_of_nodes())

    return np.mean(distances)

def second_hop_coverage(G1, G2, seeds_G1, seeds_G2, optimal_permutation):
    """
    Computes the fraction of non-seed vertices that lie within two hops
    of the seed set.

    Parameters
    ----------
    G1 : ndarray
        Adjacency matrix of Graph 1.

    seeds_G1 : array-like
        Indices of seed vertices.

    Returns
    -------
    float
        Fraction of non-seed vertices reachable within two hops of
        the seed set.
    """

    seeds = set(seeds_G1)

    reachable = set()

    for seed in seeds:

        first_neighbors = np.where(G1[seed] > 0)[0]

        for neighbor in first_neighbors:

            if neighbor not in seeds:
                reachable.add(neighbor)

            second_neighbors = np.where(G1[neighbor] > 0)[0]

            for second_neighbor in second_neighbors:
                if second_neighbor not in seeds:
                    reachable.add(second_neighbor)

    possible = G1.shape[0] - len(seeds)

    return len(reachable) / possible if possible > 0 else 0

def seed_internal_edges(G1, G2, seeds_G1, seeds_G2, optimal_permutation):
    """
    Counts the number of edges whose endpoints are both seed vertices.

    Parameters
    ----------
    G1 : ndarray
        Adjacency matrix of Graph 1.

    seeds_G1 : array-like
        Indices of seed vertices.

    Returns
    -------
    int
        Number of edges connecting pairs of seeds.
    """

    seeds = list(seeds_G1)

    internal_edges = 0

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            if G1[seeds[i], seeds[j]] > 0:
                internal_edges += 1
    print(internal_edges)
    return internal_edges


def average_distance_to_hubs(
    G1,
    G2,
    seeds_G1,
    seeds_G2,
    optimal_permutation,
    hub_percent=0.05
):
    """
    Computes the average shortest path distance from each seed to its
    nearest high-degree ("hub") vertex.

    Parameters
    ----------
    G1 : ndarray
        Adjacency matrix of Graph 1.

    seeds_G1 : array-like
        Indices of seed vertices.

    hub_percent : float
        Fraction of highest-degree vertices considered hubs.
        For example, 0.05 means the top 5% highest-degree vertices.

    Returns
    -------
    float
        Average shortest path distance from each seed to its nearest hub.
    """

    # Convert adjacency matrix to NetworkX graph
    G = nx.from_numpy_array(G1)

    degrees = np.sum(G1, axis=1)

    n_hubs = max(1, int(np.ceil(hub_percent * G1.shape[0])))

    # Indices of the highest-degree vertices
    hubs = np.argsort(degrees)[-n_hubs:]

    distances = []

    for seed in seeds_G1:

        shortest = np.inf

        for hub in hubs:

            # Distance to itself is 0 if the seed is a hub
            if seed == hub:
                shortest = 0
                break

            try:
                d = nx.shortest_path_length(
                    G,
                    source=seed,
                    target=hub
                )

                shortest = min(shortest, d)

            except nx.NetworkXNoPath:
                pass

        # Treat disconnected hubs as maximally far away
        if np.isinf(shortest):
            shortest = G.number_of_nodes()

        distances.append(shortest)

    return np.mean(distances)