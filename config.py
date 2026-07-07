
import numpy as np
# SBM Constants
N_PER_BLOCK = 200
N_BLOCKS = 3
SBM_RHO = .5
BLOCK_PROBS = np.array([
            [0.7, 0.3, 0.4],
            [0.3, 0.7, 0.3],
            [0.4, 0.3, 0.7]
        ])

#Erdos Renyi Constants
N_ER_NODES = 600
EDGE_PROBABILITY = .2
ER_RHO = .8

#Experiment Constants
TRIALS_PER_SEED_NUMBER = 25
SEED_COUNTS = [3,10, 15, 20, 50,]

# TRIALS_PER_SEED_NUMBER = 2
# SEED_COUNTS = [1,3]