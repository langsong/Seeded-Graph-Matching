# Fixed-pair seed-set classification

This experiment holds one graph pair fixed for each graph model and samples many seed sets on that pair. It isolates seed-set variation from graph-pair difficulty.

- `pairs/`: one fixed benchmark definition per graph model, including the new `paper_pa` model and the Enron pair.
- `seed_selection_common.py`: shared preparation and SGM evaluation.
- `candidate_data.py`: random and feature-diverse seed-set collection.
- `audits/solver_variance_audit.py`: repeat 200 saved historical `paper` seed sets with five solver RNG seeds.
- `audits/analyze_solver_variance.py`: quantify run-to-run variation and compare single-run and averaged labels.
- `audits/SOLVER_VARIANCE_AUDIT.md`: findings from the completed Pareto–Chung–Lu audit.
- `SEED_SET_SELECTION_FINDINGS.md`: short comparison of the four saved graph-model datasets.
- `seed_set_classification.ipynb`: classification and ranking analysis.
- `data/`: fixed graph pairs, collected seed sets, and metadata; `data/solver_variance/` holds the audit outputs.

Run commands from the repository root:

```bash
python -m fixed_pair_seed_set_classification.candidate_data --graph-model all
python -m fixed_pair_seed_set_classification.candidate_data --graph-model paper
python -m fixed_pair_seed_set_classification.candidate_data --graph-model paper_pa
python -m fixed_pair_seed_set_classification.candidate_data --graph-model enron
```

Existing datasets are protected; add `--overwrite` only when intentionally regenerating them.

The completed solver-variance diagnostic is separate from the main collection workflow. Its scripts are `python -m fixed_pair_seed_set_classification.audits.solver_variance_audit` to collect runs and `python -m fixed_pair_seed_set_classification.audits.analyze_solver_variance` to summarize saved runs; its outputs are already stored in `data/solver_variance/`.

The new `paper_pa` preset uses 600 vertices, attachment weights `alpha=0, beta=1`, ER probability `p=0.005`, ER-edge correlation `rho=0.62`, and 20 seeds. It writes `data/paper_pa_problem.npz` and `data/paper_pa_seed_sets.csv`. Set `GRAPH_MODEL = "paper_pa"` in `seed_set_classification.ipynb` to analyze it. In a small pilot on this fixed pair, 7 of 12 uniformly random seed sets reached 90% final unseeded accuracy with the usual single-run SGM settings; this is a pilot, not the expected rate for the full dataset. The older `paper` option and saved results use the Pareto-weight Chung–Lu generator, now defined in `pairs/pareto_chung_lu_seed_selection.py`; they are preserved for comparison. The solver-variance audit uses this older pair, not `paper_pa`.

The Enron command reads `utils/enron.mat`, makes each week undirected, removes loops, and retains the 129 vertices active in both weeks. This shared-vertex filter uses the known identities only to construct the benchmark; seed-set predictors still come from Graph A alone. Graph B is permuted with a fixed seed; the true mapping is saved in `data/enron_problem.npz`, and the original one-based vertex IDs are recorded in its metadata. The preset uses 25 matched seeds and the notebook's 90% success threshold; small random-seed-set pilots gave roughly half successes. To inspect its results, set `GRAPH_MODEL = "enron"` in `seed_set_classification.ipynb` and run the cells from the top.

For a custom Enron seed set, run `python -m fixed_pair_seed_set_classification.pairs.enron_seed_selection --seeds ...` with exactly 25 internal Graph-A indices from 0 to 128; the metadata maps them to the original Enron row IDs.
