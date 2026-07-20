# Contains a list of tests to run

from utils.Graphs import *
from utils.SeedingMethods import *
from test_seeding import graspologic_algorithm

TESTS = [
    {
        "name": "Dense ER Graph with SGM",
        "graph_gen_func": lambda: gen_ER_graphs(n=600, p=.2, rho=.5),
        "seeding_funcs": [random_seeds, betweenness_seeds, triangle_degree_ratio_seeds, highest_degree_seeds],
        "algorithm": graspologic_algorithm,
        "seed_numbers": [2,4,6,8,10,15,30],
        "trials": 20
    }
]