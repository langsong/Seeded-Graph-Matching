"""Real graph pairs (C. elegans, Wikipedia, Enron) for seeded graph matching experiments.

See README.md. Quick start:

    from real_data_experiments.datasets import get_pair
    A, B, truth = get_pair("worm").for_matching(shuffle_seed=0)
"""
