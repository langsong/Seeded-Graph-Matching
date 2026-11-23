# Retired experiments

Finished experiments, kept so they can be rerun and their notebooks reopened. Each still reads and writes its
original folder in `../data/`. Run one from the repository root with
`python -m selection_rule_comparison.archive.<name>`. Their docstrings and `../docs/experiment_log.md` still show the
old form, `python -m selection_rule_comparison.<name>`.

| Experiment (dates) | Files here | Data (`../data/`) | Project note |
|---|---|---|---|
| Round 1 rule comparison (20–21 Oct) | `round1_rule_comparison.ipynb` (was `rule_comparison.ipynb`), `logs/main_*`, `logs/pilot_*`; the runner is `../run_comparison.py` | `round1/`, `round1_training_budgets/` | round1-results-10-pairs-2026-10-20, round1-results-20-pairs-2026-10-21 |
| New-rules pilot (21 Oct) | `pilot_new_rules.py`, `new_rules_pilot.ipynb` | `pilot_new_rules/` | pilot-new-rules-results-2026-10-21 |
| Follow-ups (27 Oct) | `hub_test`, `borderline_check`, `regime_sweep`, `real_graph_check`, `temperature_audit` (.py and .ipynb) | `hub_test/`, `borderline/`, `regime_sweep/`, `real_graph/`, `temperature_audit/` | `../docs/experiment_log.md` |
| Beyond top degree (28 Oct) | `beyond_top_degree.py`, `.ipynb` | – | `../docs/experiment_log.md` |
| Best of N (2 Nov) | `best_of_n.py`, `.ipynb` | `best_of_n/` | weekend-runs-early-look-2026-11-07 |
| Score transfer (2 Nov) | `score_transfer.py`, `.ipynb` | `score_transfer/` (also read by `graded_outcomes.known_pair_seeds`) | `../docs/experiment_log.md` |
| SBM rules (2–3 Nov) | `sbm_rules.py`, `sbm_graphs.py`, `sbm_rules.ipynb` | `sbm_rules/` | `../docs/experiment_log.md` |
| Graded outcomes (4–5 Nov) | `graded_outcomes.ipynb` (the module stays in `../` as shared helpers) | `graded_outcomes/` | `../docs/experiment_log.md` |
| Weekend queue (7 Nov) | `weekend_queue.py`, `logs/weekend/` | – | weekend-runs-2026-11-07 |

`fixed_pair_seed_set_classification/sbm_collection.py` and `sbm_models.ipynb` import the SBM, best-of-N,
beyond-top-degree and score-transfer modules from here.
