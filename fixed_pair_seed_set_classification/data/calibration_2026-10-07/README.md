# Floor-free Chung-Lu calibration (7 Oct 2026)

This calibration uses the updated fixed-quantile Pareto Chung-Lu generator
with `alpha=2`, correlation `rho=0.70`, and no uniform ER floor. Success is
final unseeded accuracy at least 0.90. The graph-pair seeds are `3165569293`,
`18472911`, and `37051993`.

## SGM

The SGM regime uses weight scale `1.9260087733615983`, whose expected mean
degree is 5.7. The three sampled Graph-A mean degrees were 5.42, 5.71, and
5.79. Every seed set received five solver runs at `max_iter=100`; the practical
outcome is the run with the largest objective.

An eight-set-per-pair screen over k = 16, 19, 22, 25, and 28 put the pooled
best-objective success transition at k = 16. Independent confirmation used 16
new uniform-random sets per pair at k = 16 and k = 17:

| k | Pair 1 | Pair 2 | Pair 3 | Pooled best-objective | Single-run pooled | Selection misses |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 50.0% | 18.8% | 50.0% | 39.6% (19/48) | 18.8% | 0/48 |
| **17** | **43.8%** | **50.0%** | **75.0%** | **56.3% (27/48)** | **28.8%** | **0/48** |

Use **k = 17** for the new SGM collection. Its pooled 95% Wilson interval is
42.3%--69.3%; the pair-specific rates should still be reported.

## Percolation

The percolation regime uses weight scale `4.076194702720809`, whose expected
mean degree is 12. The sampled Graph-A mean degrees were 11.80, 11.71, and
11.66. Each budget used 20 uniform-random sets per pair and one tie-break run.

| k | Pair 1 | Pair 2 | Pair 3 | Pooled success | Mean accuracy |
|---:|---:|---:|---:|---:|---:|
| 3 | 15% | 5% | 0% | 6.7% | 0.146 |
| 4 | 15% | 30% | 10% | 18.3% | 0.291 |
| **5** | **50%** | **50%** | **35%** | **45.0% (27/60)** | **0.544** |
| 6 | 65% | 60% | 45% | 56.7% (34/60) | 0.649 |
| 7 | 80% | 60% | 55% | 65.0% | 0.725 |
| 8 | 95% | 85% | 85% | 88.3% | 0.889 |

Use **k = 5**: it is closest to 50%, and its pair-specific rates are reasonably
stable. Its pooled 95% Wilson interval is 33.1%--57.5%.

Raw files were produced by `audits/calibrate_chung_lu_update.py` and are kept
in this directory.
