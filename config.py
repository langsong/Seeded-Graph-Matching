
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
ER_RHO = .6

#Power Law Constants
N_PL_NODES = 600
ALPHA = 2.5
PL_RHO = .8

# PAPER Constants
N_PPR_NODES = 600
PPR_ALPHA = 2.5
PPR_EDGE_PROBABILITY = .001
PPR_RHO = .8

#Experiment Constants
TRIALS_PER_SEED_NUMBER = 25
SEED_COUNTS = [3,4,5,6,7,8,9,10,11,12,13,14,15]

np.random.seed(1234)

# Random Forest Regressor
RF_MODEL = None