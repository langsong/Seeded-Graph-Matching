import time

import numpy as np
def graph_match_percolation(
    A,
    B,
    seeds,
    r=2,
    ExpandWhenStuck=True
):
    """
    Parameters
    A : int array
        Adjacency matrix of graph A.
    B : int array
        Adjacency matrix of graph B.
    seeds : np.ndarray
        Shape (k,2). Each row [i,j] means vertex i in A corresponds to vertex j in B.
    r : int
        Minimum number of supporting matched neighbors required.
    ExpandWhenStuck : bool
        Whether to use ExpandWhenStuck.

    Returns
    -------
    dict
        corr_A, corr_B, match_order, seeds
    """
    A = np.asarray(A)
    B = np.asarray(B)
    n = A.shape[0]

    seeds = np.asarray(seeds, dtype=int)

    seeds_original = seeds.copy()

    # Current matches
    Z = seeds.copy()

    # Mark matrix
    M = np.zeros((n,n), dtype=np.int64)
    used = np.zeros((n,n), dtype=bool)
    matched_neighbors = np.zeros((n,n), dtype=bool)


    # Precompute adjacency lists
    neighbors_A = [
        np.flatnonzero(A[i])
        for i in range(n)
    ]

    neighbors_B = [
        np.flatnonzero(B[i])
        for i in range(n)
    ]

    # Precompute degrees for tie-breaking
    deg_A = np.sum(A, axis=1)
    deg_B = np.sum(B, axis=1)


    # Remove seed vertices from consideration
    for a,b in seeds:
        matched_neighbors[np.ix_(neighbors_A[a], neighbors_B[b])] = True
        M[a,:] = -n*n
        M[:,b] = -n*n


    current_seeds = seeds.copy()


    while len(current_seeds) > 0:
        # Mark neighbors of seeds
        for a_seed, b_seed in current_seeds:
            if used[a_seed, b_seed]:
                continue
            used[a_seed, b_seed] = True

            A_adj = neighbors_A[a_seed]
            B_adj = neighbors_B[b_seed]

            if len(A_adj) == 0 or len(B_adj) == 0:
                continue

            # Binary graphs:
            # every neighbor pair gets +1
            M[np.ix_(A_adj, B_adj)] += 1



        # Greedy percolation
        while np.max(M) >= r:
            max_score = np.max(M)
            candidates = np.argwhere(M == max_score)

            # Tie breaking using degree difference
            if len(candidates) > 1:
                degree_diff = (
                    np.abs(
                        deg_A[candidates[:,0]]
                        -
                        deg_B[candidates[:,1]]
                    )
                )

                best = np.where(
                    degree_diff == np.min(degree_diff)
                )[0]

                chosen = candidates[
                    np.random.choice(best)
                ]

            else:
                chosen = candidates[0]


            a_match, b_match = chosen

            # Propagate marks from new match
            A_adj = neighbors_A[a_match]
            B_adj = neighbors_B[b_match]

            matched_neighbors[np.ix_(A_adj, B_adj)] = True
            if not used[a_match, b_match] and len(A_adj) > 0 and len(B_adj) > 0:
                M[np.ix_(A_adj,B_adj)] += 1
            used[a_match, b_match] = True


            # Remove matched vertices
            M[a_match,:] = -n*n
            M[:,b_match] = -n*n


            # Add match
            Z = np.vstack(
                (
                    Z,
                    [a_match,b_match]
                )
            )
            
        # ExpandWhenStuck
        if not ExpandWhenStuck:
            break


        current_seeds = np.argwhere(
            matched_neighbors &
            ~used &
            (M > 0) &
            (M < r)
        )



    # Sort matches by A vertex
    order = np.argsort(Z[:,0])

    Z = Z[order]

    # Convert correspondence pairs into permutation format
    # perm[i] = vertex in B matched to vertex i in A
    perm = np.full(n, -1, dtype=int)

    perm[Z[:,0]] = Z[:,1]

    return perm

    # original returns, might be useful for debugging
    # return {
    #     "corr_A": Z[:,0],
    #     "corr_B": Z[:,1],
    #     "match_order": order,
    #     "seeds": seeds_original
    # }


def run_percolation_once(
    adjacency_1: np.ndarray,
    adjacency_2: np.ndarray,
    true_permutation: np.ndarray,
    seeds_1: np.ndarray,
    algorithm_seed: int,
    threshold: int = 2,
    expand_when_stuck: bool = True,
) -> dict[str, float | int | bool]:
    """Run percolation once and score unmatched nonseeds as incorrect."""

    seeds_1 = np.asarray(seeds_1, dtype=int)
    seeds_2 = true_permutation[seeds_1]
    partial_match = np.column_stack((seeds_1, seeds_2))

    random_state = np.random.get_state()
    np.random.seed(algorithm_seed)
    started = time.perf_counter()
    try:
        predicted = graph_match_percolation(
            adjacency_1,
            adjacency_2,
            partial_match,
            r=threshold,
            ExpandWhenStuck=expand_when_stuck,
        )
    finally:
        runtime_seconds = time.perf_counter() - started
        np.random.set_state(random_state)

    predicted = np.asarray(predicted, dtype=int)
    n = adjacency_1.shape[0]
    if predicted.shape != (n,):
        raise RuntimeError("Percolation returned an invalid permutation shape.")
    matched_targets = predicted[predicted >= 0]
    if np.any(matched_targets >= n) or len(np.unique(matched_targets)) != len(
        matched_targets
    ):
        raise RuntimeError("Percolation returned invalid or repeated matches.")
    if not np.array_equal(predicted[seeds_1], seeds_2):
        raise RuntimeError("Percolation did not preserve the supplied seed matches.")

    unseeded = np.setdiff1d(np.arange(n), seeds_1)
    unseeded_prediction = predicted[unseeded]
    matched = unseeded_prediction >= 0
    correct = unseeded_prediction == true_permutation[unseeded]
    incorrect = matched & ~correct
    matched_count = int(np.sum(matched))

    return {
        "final_unseeded_accuracy": float(np.mean(correct)),
        "final_all_accuracy": float(np.mean(predicted == true_permutation)),
        "percolation_matched_unseeded_fraction": float(np.mean(matched)),
        "percolation_unseeded_precision": (
            float(np.sum(correct) / matched_count) if matched_count else 0.0
        ),
        "percolation_incorrect_unseeded_fraction": float(np.mean(incorrect)),
        "percolation_matches_added": matched_count,
        "percolation_completed": bool(matched_count == len(unseeded)),
        "percolation_runtime_seconds": runtime_seconds,
    }
