# Contains a list of tests to run

from utils.Graphs import *
from utils.SeedingMethods import *
from test_seeding import graspologic_algorithm
from ExpandWhenStuck import graph_match_percolation

TESTS = [
    {
        "name": "Sparse ER Graph with SGM",
        "graph_gen_func": lambda: gen_ER_graphs(n=600, p=.01, rho=.5),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [2,5,10,15,30,50],
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
        "name": "Sparse ER Graph with Percolation",
        "graph_gen_func": lambda: gen_ER_graphs(n=600, p=.01, rho=.5),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
        "algorithm": graph_match_percolation,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    }
]

OLD_TESTS = [
    {
        "name": "Dense ER Graph with SGM",
        "graph_gen_func": lambda: gen_ER_graphs(n=600, p=.2, rho=.5),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    }
]