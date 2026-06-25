# This still needs cleaning up, optimzing, etc.
import numpy as np
# A and B are numpy adjacency matrices, both (n,n)
# seeds is (k,2) where each element is [i,j], a vertex i from A
# corresponding to a vertex j from B 
def cal_mark(x, y):
    """
    R equivalent:
        1 - abs(x-y)/max(abs(x),abs(y))
    """
    return 1 - np.abs(x-y) / np.maximum(np.abs(x), np.abs(y))


def graph_match_percolation(
        A,
        B,
        seeds,
        similarity=None,
        r=2,
        ExpandWhenStuck=False
):
    n = max(A[0].shape[0], B[0].shape[0]) # num vertices (of the larger graph)

    nc = len(A)

    # copy seeds
    seeds_ori = seeds.copy()

    Z = seeds.copy()   # matched nodes

    ns = len(seeds)

    # mark matrix
    M = np.zeros((n,n))


    # initialize similarity
    if similarity is not None:
        if np.sum(np.abs(similarity)) != 0:
            M[:, :] = similarity


    # remove seed rows/columns from consideration
    for a,b in seeds_ori:

        M[a,:] = -n*n
        M[:,b] = -n*n


    iter = 1


    while ns != 0 or ( # we won't use similarity in our exprmt
        similarity is not None
        and np.sum(np.abs(similarity)) != 0
        and iter == 1
    ):
        # MARK NEIGHBORS
        if ns != 0:

            for ch in range(nc): # this is just 1 for our experiments
                directed = not (
                    np.array_equal(A[ch], A[ch].T)
                    and
                    np.array_equal(B[ch], B[ch].T)
                )


                for seed in seeds:

                    a_seed = seed[0]
                    b_seed = seed[1]


                    A_adj = np.where(A[ch][a_seed,:] != 0)[0] #neighbors of a_seed
                    B_adj = np.where(B[ch][b_seed,:] != 0)[0] #neighbors of b_seed

                    if len(A_adj) != 0 and len(B_adj) != 0:
                        # cal_mark is just going to give us 1
                        mark = cal_mark(
                            A[ch][a_seed,A_adj][:,None],
                            B[ch][b_seed,B_adj][None,:]
                        )

                        M[np.ix_(A_adj,B_adj)] += mark


                    # directed graphs: transpose
                    if directed:

                        A[ch] = A[ch].T
                        B[ch] = B[ch].T


                        A_adj = np.where(A[ch][a_seed,:] != 0)[0]
                        B_adj = np.where(B[ch][b_seed,:] != 0)[0]


                        if len(A_adj) != 0 and len(B_adj) != 0:

                            mark = cal_mark(
                                A[ch][a_seed,A_adj][:,None],
                                B[ch][b_seed,B_adj][None,:]
                            )

                            M[np.ix_(A_adj,B_adj)] += mark


                        A[ch] = A[ch].T
                        B[ch] = B[ch].T

        
        # MATCH PAIRS WITH SCORE >= r
        while np.max(M) >= r:


            max_val = np.max(M)

            max_ind = np.argwhere(M == max_val)


            # tie-breaking
            if len(max_ind) != 1:
                # among the pairs with the highest score select the unmatched pair [i,j]
                # with the minimum degree difference
                degree_diff = np.zeros(len(max_ind))

                for ch in range(nc):
                    degree_diff += np.abs(
                        np.sum(A[ch],axis=1)[max_ind[:,0]]
                        -
                        np.sum(B[ch],axis=1)[max_ind[:,1]]
                    )
                best = np.where(
                    degree_diff == np.min(degree_diff)
                )[0]

                # if theres ties in degree difference, choose randomly
                max_ind = max_ind[
                    np.random.choice(best)
                ]
            else:
                max_ind = max_ind[0]


            a_match = max_ind[0]
            b_match = max_ind[1]

            # UPDATE MARK MATRIX
            for ch in range(nc):

                directed = not (
                    np.array_equal(A[ch], A[ch].T)
                    and
                    np.array_equal(B[ch], B[ch].T)
                )


                A_adj = np.where(
                    A[ch][a_match,:] != 0
                )[0]

                B_adj = np.where(
                    B[ch][b_match,:] != 0
                )[0]


                if len(A_adj)!=0 and len(B_adj)!=0:

                    mark = cal_mark(
                        A[ch][a_match,A_adj][:,None],
                        B[ch][b_match,B_adj][None,:]
                    )

                    M[np.ix_(A_adj,B_adj)] += mark



                if directed:
                    A[ch] = A[ch].T
                    B[ch] = B[ch].T
                    A_adj = np.where(
                        A[ch][a_match,:] != 0
                    )[0]
                    B_adj = np.where(
                        B[ch][b_match,:] != 0
                    )[0]
                    if len(A_adj)!=0 and len(B_adj)!=0:
                        mark = cal_mark(
                            A[ch][a_match,A_adj][:,None],
                            B[ch][b_match,B_adj][None,:]
                        )
                        M[np.ix_(A_adj,B_adj)] += mark
                    A[ch] = A[ch].T
                    B[ch] = B[ch].T



            # remove matched vertices
            M[a_match,:] = -n*n
            M[:,b_match] = -n*n


            # append match
            Z = np.vstack(
                [
                    Z,
                    np.array([[a_match,b_match]])
                ]
            )

        # EXPAND WHEN STUCK
        iter += 1
        ns = 0


        if ExpandWhenStuck:

            seeds_old = seeds.copy()


            seeds = np.argwhere(
                (M > 0) &
                (M < r)
            )


            ns = len(seeds)

            if np.array_equal(seeds, seeds_old):
                break



    # RETURN MATCHING
    order = np.argsort(Z[:,0])
    corr = Z[order]
    return {
        "corr_A": corr[:,0],
        "corr_B": corr[:,1],
        "match_order": order,
        "seeds": seeds_ori
    }
