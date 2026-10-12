# SGM seed-budget calibration (2026-10-06)

This calibration uses the existing sparse ER and PAPER graph parameters, three
fixed graph pairs per model, five SGM runs per seed set, `max_iter=100`, four
BLAS threads, and success at final unseeded accuracy at least 0.90.  The
practical outcome is the run with the largest SGM objective among the five.

The stage-one grid used eight nested uniform-random seed sets per pair and
candidate budget.  Its pooled objective-selected success fractions were:

| Model | Candidate budgets and success fractions |
|---|---|
| sparse ER | k=24: 0.375; k=25: 0.375; k=26: 0.542; k=27: 0.708; k=28: 0.750 |
| PAPER | k=11: 0.333; k=12: 0.417; k=13: 0.417; k=14: 0.542; k=15: 0.708 |

Independent confirmation increased the final samples to 24 seed sets per pair
for ER k=26 and 20 seed sets per pair for PAPER k=13 and k=14.

| Model | k | Pair 1 | Pair 2 | Pair 3 | Pooled | Single-run pooled |
|---|---:|---:|---:|---:|---:|---:|
| sparse ER | 26 | 0.250 | 0.958 | 0.458 | 0.556 | 0.314 |
| PAPER | 13 | 0.350 | 0.750 | 0.500 | 0.533 | 0.237 |
| PAPER | 14 | 0.850 | 0.750 | 0.550 | 0.717 | 0.367 |

The pooled 95% Wilson intervals for objective-selected success are 0.441--0.665
for ER k=26, 0.409--0.654 for PAPER k=13, and 0.592--0.815 for PAPER k=14.
No selection misses occurred: whenever any of the five runs succeeded, the run
with the largest objective also succeeded.

ER k=26 and PAPER k=13 are the recommended starting budgets for comparing seed
selection rules.  Both put random seed sets near 50% pooled success while
leaving room for a good rule to improve.  PAPER k=14 is already too easy in the
combined confirmation, especially on pair 1.  ER has substantial between-pair
difficulty variation, so each new pair should receive a separate pilot over a
small k grid before its analysis seed sets are sampled.

`calibration_summary.csv` contains the final per-pair and pooled summaries. The
stage-one and confirmation CSV files contain all complete run-level records.
Aborted exploratory outputs are retained under `incomplete/` and are excluded
from every reported calculation.
