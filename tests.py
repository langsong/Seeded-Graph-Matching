# Contains a list of tests to run

from utils.Graphs import *
from utils.SeedingMethods import *
from test_seeding import graspologic_algorithm
from ExpandWhenStuck import graph_match_percolation

TESTS = [
    
    
]

OLD_TESTS = [
    {
        "name": "Sparse ER Graph with SGM",
        "graph_gen_func": lambda: gen_ER_graphs(n=600, p=.01, rho=.5),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [0,10,30,50,100,150],
        "trials": 20
    },
    {
        "name": "Sparse ER Graph with Percolation",
        "graph_gen_func": lambda: gen_ER_graphs(n=600, p=.01, rho=.5),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
        "algorithm": graph_match_percolation,
        "seed_numbers": [0,10,30,50,100,150],
        "trials": 20
    },
    {
        "name": "Dense ER Graph with SGM",
        "graph_gen_func": lambda: gen_ER_graphs(n=600, p=.2, rho=.5),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    },
    {
        "name": "Dense ER Graph with Percolation",
        "graph_gen_func": lambda: gen_ER_graphs(n=600, p=.2, rho=.5),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
        "algorithm": graph_match_percolation,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    },
    {
        "name": "Dense SBM Graph with SGM",
        "graph_gen_func": lambda: gen_SBM_graphs(n_per_block=200, n_blocks=3, rho=.5, 
                                                 block_probs=np.array([
                                                    [0.7, 0.3, 0.4],
                                                    [0.3, 0.7, 0.3],
                                                    [0.4, 0.3, 0.7]
                                                ])),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, blocked_random_seeds, neighbor_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    },
    {
        "name": "Sparse SBM Graph with SGM",
        "graph_gen_func": lambda: gen_SBM_graphs(n_per_block=200, n_blocks=3, rho=.8, 
                                                 block_probs=np.array([
                                                    [0.02, 0.005, 0.01],
                                                    [0.005, 0.02, 0.005],
                                                    [0.01, 0.005, 0.02]
                                                ])),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, blocked_random_seeds, neighbor_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    },
    {
        "name": "Dense SBM Graph with Percolation",
        "graph_gen_func": lambda: gen_SBM_graphs(n_per_block=200, n_blocks=3, rho=.5, 
                                                 block_probs=np.array([
                                                    [0.7, 0.3, 0.4],
                                                    [0.3, 0.7, 0.3],
                                                    [0.4, 0.3, 0.7]
                                                ])),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, blocked_random_seeds, neighbor_degree_seeds],
        "algorithm": graph_match_percolation,
        "seed_numbers": [2,4,6,8,10,15,30,50],
        "trials": 20
    },
    {
        "name": "Sparse SBM Graph with Percolation",
        "graph_gen_func": lambda: gen_SBM_graphs(n_per_block=200, n_blocks=3, rho=.8, 
                                                 block_probs=np.array([
                                                    [0.02, 0.005, 0.01],
                                                    [0.005, 0.02, 0.005],
                                                    [0.01, 0.005, 0.02]
                                                ])),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, blocked_random_seeds, neighbor_degree_seeds],
        "algorithm": graph_match_percolation,
        "seed_numbers": [2,4,6,8,10,15,30,50],
        "trials": 20
    },
    {
        "name": "Heavy tail PAPER Graph with SGM",
        "graph_gen_func": lambda: gen_PAPER_graphs(n=600, alpha=2, p=.005, rho=.7),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds, neighbor_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    },
    {
        "name": "Light tail PAPER Graph with SGM",
        "graph_gen_func": lambda: gen_PAPER_graphs(n=600, alpha=3, p=.005, rho=.7),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds, neighbor_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    },
    {
        "name": "Heavy tail PAPER Graph with Percolation",
        "graph_gen_func": lambda: gen_PAPER_graphs(n=600, alpha=2, p=.005, rho=.7),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds, neighbor_degree_seeds],
        "algorithm": graph_match_percolation,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    },
    {
        "name": "Light tail PAPER Graph with Percolation",
        "graph_gen_func": lambda: gen_PAPER_graphs(n=600, alpha=3, p=.005, rho=.7),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds, neighbor_degree_seeds],
        "algorithm": graph_match_percolation,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    }
]