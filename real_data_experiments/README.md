# Real-network experiments

The seed-selection questions on real graph pairs: the C. elegans connectome (chemical vs electrical, hermaphrodite and
male), English vs French Wikipedia, and Enron email weeks. Raw files are in `../Data/`; `prepare_data.py` writes the
prepared pairs to `data/`.

Run from the repository root in the `ds` environment, e.g. `python -m real_data_experiments.strategy_and_active_v2 --status`.

| Experiment (date) | Code | Notebook | Results | Project note |
|---|---|---|---|---|
| Data preparation | `datasets.py`, `prepare_data.py`, `rdata_reader.py` | – | `data/` | – |
| First look (7 Nov) | `first_look.py` | `notebooks/01_first_look` | `results/first_look/` | real-data-notebooks-review-2026-11-17 |
| Broader candidates and selector transfer (11 Nov) | `broader_candidates.py` (was `follow_up.py`) | sections of `01_first_look` | `results/follow_up/` | real-data-notebooks-review-2026-11-17 |
| Score-range banks, real (17 Nov) | `../selection_rule_comparison/collect_score_ranges.py` | `notebooks/02_score_range_banks_real` | `../selection_rule_comparison/data/score_range_collection/real/` | real-data-notebooks-review-2026-11-17 |
| Strategy comparison v1 (17 Nov) | `strategy_comparison.py` | none (read with `strategy_comparison.read()`) | `results/strategy_comparison/` | strategy-comparison-results-2026-11-17 |
| Active test v1 (17 Nov) | `active_test.py` | none | `results/active_test/` | active-test-results-2026-11-18 |
| Strategy comparison and active queries v2 (18 Nov) | `strategy_and_active_v2.py` (was `overnight.py`), `strategy_and_active_v2_analysis.py` | `notebooks/03_strategy_comparison_v2`, `04_active_queries_v2` | `results/overnight_v2/` | overnight-v2-results-2026-11-18; `docs/strategy_and_active_v2.md` |

v2 reuses code from `strategy_comparison.py` and `active_test.py`. Their v1 notebooks were turned into the v2
notebooks on 18 Nov.

Tests: `python -m unittest real_data_experiments.tests.test_active_test real_data_experiments.tests.test_strategy_comparison real_data_experiments.tests.test_strategy_and_active_v2`.

## Moved on 23 Nov 2026
| Was | Now |
|---|---|
| `overnight.py`, `overnight_analysis.py` | `strategy_and_active_v2.py`, `strategy_and_active_v2_analysis.py` |
| `OVERNIGHT.md` | `docs/strategy_and_active_v2.md` |
| `follow_up.py` | `broader_candidates.py` |
| `README.md` | `docs/experiment_log.md` |
| `first_look.ipynb` | `notebooks/01_first_look.ipynb` |
| `score_range_analysis.ipynb` | `notebooks/02_score_range_banks_real.ipynb` |
| `strategy_comparison.ipynb` (showed v2 results) | `notebooks/03_strategy_comparison_v2.ipynb` |
| `active_test.ipynb` (showed v2 results) | `notebooks/04_active_queries_v2.ipynb` |
| `test_overnight.py`, `test_active_test.py`, `test_strategy_comparison.py` | `tests/test_strategy_and_active_v2.py`, `tests/test_active_test.py`, `tests/test_strategy_comparison.py` |

Result folders kept their names (`results/overnight_v2/`, `results/follow_up/`). `strategy_and_active_v2.py`
checks its code hashes on resume, so it will refuse to resume `results/overnight_v2/`; give new runs a new
`--output-dir`.
