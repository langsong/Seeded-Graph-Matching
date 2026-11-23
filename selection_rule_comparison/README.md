# Seed-selection experiments on synthetic graphs

Which k seeds should we buy so that FAQ, GOAT or percolation matching does best, and when does the choice matter?
Everything here uses synthetic pairs (ER, Chung–Lu, PAPER). Real networks are in `../real_data_experiments/`.

Run from the repository root in the `ds` environment: `python -m selection_rule_comparison.<module> --help`.

## Current experiments
| Experiment (dates) | Question | Code | Notebook | Data | Project note |
|---|---|---|---|---|---|
| Fresh pairs (3–11 Nov) | Does the frozen score track accuracy on new pairs, and how do constructed sets do over redraws of B? | `fresh_pair_scores.py`, `fresh_pair_extremes.py`, `b_replicates.py`, `constructed_redraws.py` | `notebooks/01_fresh_pair_scores`, `05_constructed_redraws` | `data/fresh_pair_scores/` | fresh-pair-extremes-findings-2026-11-11, constructed-redraws-2026-11-11 |
| Regime map (6 Nov) | Across density, ρ and budget, where are outcomes graded, and where does seed choice matter? | `regime_map.py` | `02_regime_map` | `data/regime_map/` | weekend-runs-findings-2026-11-08 |
| Landscape (7–11 Nov) | In graded cells, how does match ratio track the seed-set features, and which rules win? | `landscape.py`, `landscape_extremes.py` | `03_landscape` | `data/landscape/`, `data/landscape_extremes/` | weekend-runs-early-look-2026-11-07, project-review-2026-11-10 |
| Learned selector (10 Nov) | Does a score fitted where good sets live beat the best hand-made rule? | `learned_selector.py` | `04_learned_selector` | `data/learned_selector/` | learned-selector-2026-11-10, learned-selector-results-2026-11-10 |
| Score-range banks (17 Nov) | Frozen score against accuracy over the full score range (150 sets per bank) | `collect_score_ranges.py`, `score_range_analysis.py` | `06_score_range_banks` (real banks: `../real_data_experiments/notebooks/02_score_range_banks_real`) | `data/score_range_collection/`, `data/score_range_analysis/` | real-data-notebooks-review-2026-11-17; `docs/score_range_collection.md` |
| 11 draws of B (21–22 Nov) | The same banks, each set averaged over 11 draws of B | `score_range_draws.py`, `score_range_draws_analysis.py` | `07_score_range_11_draws` | `data/score_range_draws/` | score-range-11-draws-results-2026-11-22; `docs/score_range_draws.md` |

Project notes are in the claude.ai project "Seeded Graph Matching" (`claude/<name>.md`).

## Shared code
- `settings.py`: graph and solver settings.
- `strategies.py`: the hand-made rules (orderings of A's vertices) and seed-set features.
- `run_comparison.py`: seed streams, graph hashing and single runs used everywhere, plus the round-1 runner (its notebook is in `archive/`).
- `graded_outcomes.py`: collection helpers (row writer, resume, set drawing, pair seeds already used) used by most collectors and by `real_data_experiments`; also the 4–5 Nov graded-outcomes experiment (notebook in `archive/`).
- `b_replicates.py`: draws a new B given A (`draw_b`), used by the landscape, learned selector and score-range collectors; also the fresh-pair redraws.
- `jobs.py`: the parallel task runner (moved unchanged from `sbm_rules.py`).
- `pooled_weights.py`: loads the frozen pooled weights (moved unchanged from `best_of_n.py`).

## Other folders
- `notebooks/`: current notebooks, numbered in the order the work was done.
- `tests/`: `python -m unittest selection_rule_comparison.tests.test_layout selection_rule_comparison.tests.test_score_range_draws`.
- `docs/`: `experiment_log.md` (the dated log that was this README until 23 Nov), plus the score-range design notes.
- `archive/`: retired experiments, still runnable. See `archive/README.md`.
- `data/`: results, one folder per experiment. See `data/README.md`. No data folder moved.

## Moved on 23 Nov 2026
| Was | Now |
|---|---|
| `README.md` | `docs/experiment_log.md` |
| `SCORE_RANGE_COLLECTION.md`, `SCORE_RANGE_DRAWS.md` | `docs/score_range_collection.md`, `docs/score_range_draws.md` |
| `fresh_pair_scores.ipynb`, `regime_map.ipynb`, `landscape.ipynb`, `learned_selector.ipynb`, `constructed_redraws.ipynb` | `notebooks/01_` … `05_` with the same names |
| `score_range_analysis.ipynb` | `notebooks/06_score_range_banks.ipynb` |
| `score_range_11_draws.ipynb` | `notebooks/07_score_range_11_draws.ipynb` |
| `test_score_range_draws.py` | `tests/test_score_range_draws.py` |
| 12 retired modules, 12 notebooks, `logs/` | `archive/` (`rule_comparison.ipynb` is now `archive/round1_rule_comparison.ipynb`) |
| `run_tasks` in `sbm_rules.py`; `load_weights` in `best_of_n.py` | `jobs.py`; `pooled_weights.py` |

**Finished collections cannot be resumed.** `score_range_draws.py` and `collect_score_ranges.py` compare their code
with the hashes saved in their finished folders, and some of the files they hash changed (imports only). They will
refuse to resume `data/score_range_draws/` or `data/score_range_collection/`, so give new runs a new `--output-dir`.
