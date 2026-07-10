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

    """
    A = np.asarray(A)
    B = np.asarray(B)
    n = A.shape[0]

    seeds = np.asarray(seeds, dtype=int)

    seeds_original = seeds.copy()

    # Current matches
    Z = seeds.copy()

    # Mark matrix
    M = np.zeros((n,n), dtype=np.int16)


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
        M[a,:] = -n*n
        M[:,b] = -n*n


    current_seeds = seeds.copy()


    while len(current_seeds) > 0:
        # Mark neighbors of seeds
        for a_seed, b_seed in current_seeds:

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

            if len(A_adj) > 0 and len(B_adj) > 0:
                M[np.ix_(A_adj,B_adj)] += 1


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


        old_seeds = current_seeds.copy()

        current_seeds = np.argwhere(
            (M > 0) &
            (M < r)
        )

        # stop if expansion did nothing
        if (
            len(current_seeds) == len(old_seeds)
            and np.array_equal(
                current_seeds,
                old_seeds
            )
        ):
            break



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