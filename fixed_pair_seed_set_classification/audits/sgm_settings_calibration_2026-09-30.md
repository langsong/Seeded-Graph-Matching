# Fixed-pair SGM setting calibration (2026-09-30)

The fixed-pair benchmarks use one 600-vertex graph pair per model. The existing
presets are sparse ER `(p=0.010, rho=0.80, k=18)`, PAPER
`(ER-background p=0.005, rho=0.62, k=20)`, and fixed-weight Pareto–Chung–Lu
`(ER-background p=0.005, rho=0.70, k=30)`. PAPER also has a shared preferential-
attachment tree (`alpha=0, beta=1`); Chung–Lu uses Pareto quantile weights
(`alpha=2`). Pair seeds remain `4192889773`, `424242`, and `3165569293`,
respectively.

I screened 36 `(p, rho, k)` combinations per model at 100 iterations, using
four random seed sets and one solver RNG seed per combination. The screen only
located the transition region. I then held each model's existing `p`, `rho`,
and graph-pair seed fixed and tested 20 random seed sets with five solver RNG
seeds (2026–2030) per setting. Success means final *unseeded* accuracy at
least 0.90. The paired 100- and 150-iteration runs use the same graph, seed
sets, and solver RNG seeds.

| Model | Recommended p | rho | k | Mean degree in A | Success at 100 | Success at 150 | Labels changed from 100 to 150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse ER | 0.010 | 0.80 | 17 | 5.92 | 45/100 | 45/100 | 0 |
| PAPER | 0.005 background | 0.62 | 18 | 5.09 | 55/100 | 55/100 | 0 |
| Fixed Pareto–Chung–Lu | 0.005 background | 0.70 | 24 | 5.72 | 62/100 | 63/100 | 1 |

At 150 iterations, mean SGM runtime per run was 6.29 s (ER), 8.06 s
(PAPER), and 4.09 s (Chung–Lu). The corresponding 100-iteration times were
5.14 s, 6.11 s, and 3.94 s. For a common conservative protocol under the
**one-thread audit setting**, use `max_iter=150`; 100 iterations gave nearly
identical labels in that audit.
The 150-iteration cap was still reached by many already-successful ER and
PAPER runs, so a cap hit alone does not imply a failed match.

For comparison, at 100 iterations ER `k=16` succeeded in 36/100 runs and
`k=20` in 88/100; Chung–Lu `k=26` succeeded in 68/100. These results support
reducing the seed budget rather than changing the current edge density or
correlation for the first fixed-pair experiment. The calibration is specific
to these graph-pair realizations; do not treat the percentages as graph-family
success probabilities.

The raw screen and confirmation CSVs are in
`fixed_pair_seed_set_classification/data/calibration_2026-09-30/`. The sweep
runner is `fixed_pair_seed_set_classification/audits/calibrate_sgm_settings.py`.
This audit does not change the experiment presets or prior datasets.

## Runtime and thread-setting check

The original sweep forced one BLAS/OpenMP thread. The saved 30-iteration
datasets used about 0.98 s/run (ER) and 1.05 s/run (PAPER), whereas replaying
one of those runs now took about 2.0 s with one thread and 1.0 s with the
environment's normal four threads. I reran 20 seed sets × 5 solver seeds at
100 iterations with four threads, using identical graph pairs and seed sets.

| Model | One-thread success | Four-thread success | Different run labels | Four-thread mean runtime | Projected 1,000 sets × 5 runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse ER | 45/100 | 40/100 | 31/100 | 2.03 s | 2.82 h |
| PAPER | 55/100 | 60/100 | 39/100 | 2.56 s | 3.55 h |
| Fixed Pareto–Chung–Lu | 62/100 | 62/100 | 0/100 | 1.58 s | 2.19 h |

The graph arrays, truth permutations, and sampled seed sets have identical
hashes between thread settings; a one-thread replay reproduced the saved
one-thread results. Thus the thread setting affects some SGM trajectories and
should be held fixed and recorded. The calibrated settings remain mixed under
four threads. The projected hours are SGM-only sequential time and exclude
feature extraction; they replace the larger one-thread wall-time projection.
The effect of 150 iterations has not been retested under four threads, so
100 iterations is the economical pilot setting and 150 can be a sensitivity
check on a subset under the same thread setting.

## Nearby settings checked with four threads

The following comparisons all use the same fixed pair per model, 20 uniformly
random seed sets per setting, five solver RNG seeds per set (2026–2030),
`max_iter=100`, and four BLAS/OpenMP threads. Success is final unseeded accuracy
at least 0.90. The variance ratio is the observed variance across seed sets of
their five-run success fractions divided by the mean within-set Bernoulli
variance; it is a **ratio**, not a fraction or percentage, and is noisy with
only five repeats and 20 sets.

| Model | Parameters | k | Successes / 100 | Mixed-outcome sets / 20 | Variance ratio | Mean SGM time / run |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sparse ER | p=0.010, rho=0.80 | 17 | 40 | 14 | 0.67 | 2.03 s |
| Sparse ER | p=0.010, rho=0.70 | 27 | 25 | 11 | 0.95 | 1.50 s |
| Sparse ER | p=0.010, rho=0.70 | 30 | 42 | 13 | 0.79 | 1.60 s |
| PAPER | background p=0.005, rho=0.62 | 14 | 26 | 13 | 0.46 | 2.01 s |
| PAPER | background p=0.005, rho=0.62 | 16 | 52 | 15 | 0.73 | 2.32 s |
| PAPER | background p=0.005, rho=0.62 | 18 | 60 | 15 | 0.67 | 2.56 s |
| Fixed Pareto–Chung–Lu | background p=0.005, rho=0.70, alpha=2 | 22 | 43 | 9 | 1.55 | 1.60 s |
| Fixed Pareto–Chung–Lu | background p=0.005, rho=0.70, alpha=2 | 24 | 62 | 13 | 0.84 | 1.58 s |

For the next fixed-pair pilot, use **ER p=0.010, rho=0.70, k=30**;
**PAPER background p=0.005, rho=0.62, k=16**; and **fixed Pareto–Chung–Lu
background p=0.005, rho=0.70, alpha=2, k=22**. Keep n=600, the fixed pair
seeds above, 100 iterations, five solver RNG seeds, and four threads. These
choices yield roughly balanced random-set outcomes on the tested pairs. The ER
choice has only a modest apparent seed-set signal advantage over rho=0.80,
k=17; the latter is a useful lower-query-budget sensitivity setting. These
20-set estimates are calibration, not a guarantee for the full candidate
distribution or new graph pairs. The rows above supersede the earlier pilot
choices in this file; the earlier one-thread comparison is retained as an
audit of thread sensitivity. No experiment preset or existing dataset was
changed by this calibration.
