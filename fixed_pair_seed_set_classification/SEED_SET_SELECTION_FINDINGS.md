# Seed-set selection: four-model check

I reran the notebook on the four saved datasets. Each uses one calibrated graph pair, 900 training sets, and 100 random test sets. No SGM runs were repeated.

| Graph | Random-test success | Logistic AUC | Forest AUC | Forest top-10 successes |
| --- | ---: | ---: | ---: | ---: |
| Sparse ER | 37% | 0.829 | 0.803 | 6/10 |
| Sparse SBM | 44% | 0.720 | 0.669 | 8/10 |
| Pareto–Chung–Lu (`paper`) | 49% | 0.715 | 0.690 | 9/10 |
| IER | 48% | 0.780 | 0.756 | 7/10 |

Multi-hop coverage is the most consistent signal: three-hop seed count leads forest importance in the historical `paper` and IER datasets, two-hop/three-seed coverage in ER, and direct two-witness coverage in SBM. The two multi-hop measures correlate 0.88–0.95; signature entropy correlates 0.94–0.96 with seed degree. These results do not isolate causal effects. The newer `paper_pa` dataset is not included in this four-model comparison.

Next compare **highest-degree seeds**, **greedy three-hop coverage**, and **greedy capped one-hop witness count** (up to two seed neighbors per vertex) at equal budgets on new pairs. Treat two-hop/three-seed coverage as an alternate multi-hop score. The current top-pick results use one calibrated pair and one SGM run per set; [cross-pair results](../cross_pair_seed_set_quality/regressor.ipynb) also reflect pair difficulty, and the [solver audit](audits/SOLVER_VARIANCE_AUDIT.md) found unstable single-run labels.
