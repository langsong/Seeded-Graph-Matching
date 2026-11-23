# One-seed marginal problem

This experiment starts from an existing seed set and asks which unseeded Graph A vertex should be queried next.

- `one_seed_problem.py`: fixed graph pairs, existing seed sets, and one-candidate SGM evaluation.
- `candidate_features.py`: compact pre-query features and post-query outcomes for every candidate.
- `one_seed_diagnostics.ipynb`: regression, classification, ranking, and gradient diagnostics.
- `goat_high_rho_singletons.ipynb`: separate high-correlation GOAT study with no existing seeds. Choose the `ds` kernel and run all cells to test every singleton on the saved 600-vertex ER pair (default target mean degree 12, rho 0.70; degree 16 is optional). Results checkpoint after each call under `data/goat_high_rho_singletons/`; `MAX_VERTICES = 5` gives a short trial that can later resume to all 600. Includes a no-seed control, a small repeatability check, vertex-feature plots, and exploratory within-pair prediction.
- `one_seed_problems.py` and `candidata_selection.ipynb`: the earlier version retained for comparison.
- `data/`: graph pairs, candidate tables, and metadata.
- `claude_outputs/`: external diagnostic output retained with this experiment.

Run commands from the repository root:

```bash
python -m one_seed_marginal_problem.candidate_features --problem all
python -m one_seed_marginal_problem.one_seed_problem --problem paper --candidate 0
```

Existing candidate tables are protected; add `--overwrite` to the collection command only when intentionally regenerating them.
