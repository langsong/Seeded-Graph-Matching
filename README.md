# Seed selection for seeded graph matching

Given graph A and a budget of k correspondence queries, which seeds should we pay for so that seeded graph matching
(FAQ, GOAT) or percolation matching does best, and when does the choice matter? Started as a UMD Math REU project
(summer 2026); now a PhD thesis chapter.

Run code from this folder (the repository root) in the `ds` conda environment, as modules, for example
`python -m selection_rule_comparison.score_range_draws --status`.

## Where things are
| Folder | What it holds |
|---|---|
| `selection_rule_comparison/` | Seed-selection experiments on synthetic graphs (ER, Chung–Lu, PAPER): regime map, landscape, learned selector, fresh pairs, score-range banks. Start with its README. |
| `real_data_experiments/` | The same questions on real pairs (C. elegans, Wikipedia, Enron). |
| `fixed_pair_seed_set_classification/` | Seed-set feature models. `data/multi_pair_2026-10-30/pooled_weights.json` holds the frozen pooled weights that the scores use. |
| `experiment_common/` | Shared code: graph presets and pair generation, features, SGM (FAQ, GOAT) and percolation matching, provenance. |
| `utils/`, `config.py` | Graph generators (`utils/Graphs.py`) still used by `experiment_common` and the experiments; `config.py` is read by `utils/`. |
| `Data/` | Raw real-network files (not in git). |
| `research_notes/` | Planning notes, calibration records and weekly updates. |
| `archive/` | Finished side projects, kept as they were. See `archive/README.md`. |
| `tmp/` | Scratch space (not in git). |

## Data and git
Results sit next to the code that wrote them: `selection_rule_comparison/data/`, `real_data_experiments/results/` and
`fixed_pair_seed_set_classification/data/`. The large ones are not in git (about 1.5 GB), so back them up separately.

## Tests
```bash
python -m unittest selection_rule_comparison.tests.test_layout selection_rule_comparison.tests.test_score_range_draws \
    real_data_experiments.tests.test_active_test real_data_experiments.tests.test_strategy_comparison \
    real_data_experiments.tests.test_strategy_and_active_v2
```

## Reorganization of 23 Nov 2026
Files were moved and a few renamed; saved results did not move. Each package README lists its moves, and
`research_notes/reorganization_2026-11-23_moves.tsv` has every move. A snapshot of all code and notebooks from before
the change is in `tmp/pre_reorg_2026-11-23/`.
