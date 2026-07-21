import os
import json
from pathlib import Path
from test_seeding import *
from tests import TESTS

def run_experiment_sweep(tests):
    """
    Loops through a list of configuration dictionaries to run graph matching experiments,
    creates dedicated results folders, and plots the results.
    
    Parameters:
    config_list (list of dict): A list where each dictionary contains the keys:
        - 'name': String representing the name of the test
        - 'graph_gen_func': Function to generate the graphs
        - 'seeding_funcs': List of seeding strategy functions
        - 'algorithm': The graph matching algorithm to use
        - 'seed_numbers': List of seed counts to evaluate
        - 'trials': Number of trials per seed count
    """
    for test in tests:
        # Extract parameters from the current configuration dictionary
        test_name = test['name']
        graph_gen_func = test['graph_gen_func']
        seeding_funcs = test['seeding_funcs']
        algorithm = test['algorithm']
        seed_numbers = test['seed_numbers']
        trials = test['trials']
        
        print(f"Starting experiment: {test_name}...")
        
        # 1. Create a folder inside the 'results' directory named after the test
        save_folder = Path(f"results/{test_name}") 
        os.makedirs(save_folder, exist_ok=True)
        
        # 2. Call compare_seeding_sequential with the given parameters
        accuracies, runtimes = compare_seeding_sequential(
            graph_gen_func=graph_gen_func,
            seeding_funcs_list=seeding_funcs,
            algorithm=algorithm,
            seed_nums_list=seed_numbers,
            n_trials=trials
        )
        
        # 3. Call plot_results to save the visualization into the new folder
        plot_results(accuracies, seed_numbers, y_label="Match Ratio", show_plot=False, out_file=f"results/{test_name}/accuracy_by_seed_count.png")
        plot_results(runtimes, seed_numbers, y_label="CPU time", show_plot=False, out_file=f"results/{test_name}/runtime_by_seed_count.png")

        # 4. Save results as json files
        with open(f"results/{test_name}/accuracy_data.json", "w") as f:
            json.dump(accuracies, f)

        with open(f"results/{test_name}/runtime_data.json", "w") as f:
            json.dump(accuracies, f)


if __name__ == "__main__":
    run_experiment_sweep(TESTS)