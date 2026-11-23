# Actual graspologic FAQ and GOAT calibration, 13 October 2026

The recommended starting family is mean degree about 16, with the budgets below.
Use GOAT for the main single-run seed-quality label collection and retain FAQ as
the standard solver comparison. FAQ remains sensitive to RNG on Chung-Lu, even
after removing disconnected and structurally symmetric graphs.

These are calibration settings for `n_init=1`, not best-of-five settings.

## Recommended parameters

| Model | Density parameter | rho | FAQ seeds | FAQ success | GOAT seeds | GOAT regularizer | GOAT success |
|---|---|---:|---:|---:|---:|---:|---:|
| ER | `er_probability=16/599` (0.0267111853) | 0.50 | 29 | 17/36 (47.2%) | 12 | 100 | 19/36 (52.8%) |
| PAPER | `paper_pa_probability=0.023378`, alpha 0, beta 1 | 0.40 | 31 | 19/36 (52.8%) | 12 | 300 | 16/36 (44.4%) |
| Chung-Lu | alpha 2, scale 5.452064, floor 0 | 0.45 | 27 | 20/36 (55.6%) | 9 | 300 | 16/36 (44.4%) |

Common settings: n=600, barycenter initialization, `init_perturbation=0`,
`shuffle_input=True`, `n_init=1`, `max_iter=100`, `tol=0.01`, 4 BLAS threads.
GOAT uses `transport=True`, `transport_tol=0.05`, and the recommended
`transport_max_iter=5000`. PAPER's rho applies to its ER background; the tree
is shared exactly. PAPER's expected mean degree is about 15.953, ER's is 16,
and Chung-Lu's is 16.00004 with the actual generator's structural cutoff.

Each rate pools three graph pairs with 12 uniformly random seed sets per pair.
At 36 sets these are pilot estimates, with sampling uncertainty roughly
15 percentage points around rates near 50%. Four sets per pair were used for
the initial budget screen and were reused in confirmation; eight further sets
per pair were then added. These are not held-out performance estimates.

## Suggested initial fixed pairs

The following saved pairs gave reasonably balanced outcomes for both solvers.
They are explicitly selected from the calibration for a fixed-pair study, so
new seed sets must be used for subsequent collection and model evaluation.

| Model | Graph-pair RNG seed | Saved pair ID | FAQ successes | GOAT successes |
|---|---:|---:|---:|---:|
| ER | 88424077 | 2 | 8/12 (66.7%) | 7/12 (58.3%) |
| PAPER | 7301002 | 2 | 7/12 (58.3%) | 5/12 (41.7%) |
| Chung-Lu | 18472911 | 1 | 4/12 (33.3%) | 7/12 (58.3%) |

All nine generated pairs passed connectivity and discrete 1-WL color
refinement, with zero rejected draws. A discrete refinement certifies no
non-identity automorphism; a nondiscrete refinement would be inconclusive.
No edge-retention or hub-size acceptance filter was applied.

Freeze the selected graph pairs, density, rho, solver settings, and k before
collecting the main data. Retain the other pairs for replication with the same
budgets rather than changing k to make every pair have the same success rate.

## Pair variability

| Model / solver | Pair 0 | Pair 1 | Pair 2 |
|---|---:|---:|---:|
| ER / FAQ, k=29 | 6/12 | 3/12 | 8/12 |
| ER / GOAT, k=12 | 9/12 | 3/12 | 7/12 |
| PAPER / FAQ, k=31 | 7/12 | 5/12 | 7/12 |
| PAPER / GOAT, k=12 | 10/12 | 1/12 | 5/12 |
| Chung-Lu / FAQ, k=27 | 11/12 | 4/12 | 5/12 |
| Chung-Lu / GOAT, k=9 | 8/12 | 7/12 | 1/12 |

The pooled class balance conceals substantial differences, particularly for
PAPER GOAT and Chung-Lu. The PAPER graph-A maximum degrees were 71, 37, and 51;
the lowest-hub pair was the hardest for GOAT. This is an association in three
draws, not a causal finding. Pair metadata record realized edge correlation,
retained-edge fraction, degree statistics, and structural checks.

## RNG audit

Three seed sets per pair were evaluated over three independent solver RNG seeds
at the selected budgets: nine audited sets per model and solver. Pairwise label
agreement averages the three pairwise comparisons within each seed set.

| Model | FAQ pairwise label agreement | FAQ mixed-label sets | GOAT pairwise label agreement | GOAT mixed-label sets |
|---|---:|---:|---:|---:|
| ER | 92.6% | 1/9 | 100% | 0/9 |
| PAPER | 85.2% | 2/9 | 100% | 0/9 |
| Chung-Lu | 63.0% | 5/9 | 100% | 0/9 |

At the initial 1000-iteration transport limit, GOAT returned identical full
matchings and accuracies for every audited set across RNG seeds. The audited
warning cases were checked again at 5000 iterations over the other RNG seeds;
their accuracies and labels also agreed exactly. The PAPER set whose label was
corrected by a higher inner limit also succeeded across three RNG seeds.

This supports one-run GOAT labels in this family. FAQ ER looks more stable;
FAQ PAPER and especially Chung-Lu require repeated evaluation observations.
For a fair single-run solver comparison, keep FAQ at `n_init=1`, and preserve
repeat outcomes as an audit or success-fraction estimate. Selecting the best
objective among repeats would be a different algorithm and require separate
budget calibration.

## Numerical checks and correction

The main sweep initially used graspologic's default `transport_max_iter=1000`.
Eight GOAT runs at the selected settings emitted Sinkhorn convergence warnings:
one ER and seven PAPER runs. All eight were rerun with `transport_max_iter=5000`.
Warnings disappeared in all eight, and one PAPER run changed from 28.4% accuracy
to 100%. Therefore PAPER GOAT's corrected success rate is 16/36 rather than
15/36. The recommended rates above incorporate this correction.

For runs that already stopped Sinkhorn before 1000 iterations, raising its
iteration cap leaves the algorithm's computed path unchanged. These runs were
retained; only the eight capped cases needed rerunning. `recommended_runs.csv`
marks each row's source and keeps the original accuracy and label.

Two selected GOAT runs reached the 100-iteration outer cap: one successful ER
run and one unsuccessful Chung-Lu run. Raising the outer cap to 200 changed
their accuracies from 0.9915 to 1.000 and 0.3283 to 0.3503 respectively, without
changing either success label. No selected soft solution contained nonfinite
values. Corrected selected runs had no Sinkhorn warnings.

Every corrected selected failure was below 0.383 accuracy and every success
was at least 0.977, so the 0.90 cutoff cleanly separates these pilot outcomes.

## Reproducibility and files

The sweep used the public `graspologic.match.graph_match` API, graspologic 3.4.4,
POT 0.9.7, NumPy 1.26.4, and SciPy 1.17.1. It used project generators directly,
not copies. Thread settings were fixed throughout. FAQ can change outcomes
with different floating-point implementations or thread counts even at a fixed
RNG seed, so keep the environment and thread limit consistent when reproducing.

The 480 main runs comprise a 216-run budget screen, 156 added confirmation runs,
and 108 extra RNG audit runs. There were 16 additional numerical and corrected
RNG checks. Main solver time was about 12 minutes on this machine; individual
runs typically took about 1 second for FAQ and 2 seconds for GOAT.

- `runs.csv`: all 480 original runs, including seed vertices, solver RNG seeds,
  final permutations, objective, convergence, runtime, warnings, and soft-matrix
  diagnostics.
- `recommended_runs.csv`: 216 primary outcomes at the selected budgets after
  correcting the eight transport-cap cases. Revised permutations are not
  included because the numerical probes stored accuracy and objective only.
- `recommended_summary_5000.csv` and `recommended_pair_summary.csv`: corrected
  rates used in this report.
- `rng_audit_summary.csv` and `rng_audit_sets.csv`: original repeated-run
  diagnostics. The variance summary distinguishes pair effects, variation
  between seed sets within pairs, and RNG variation.
- `numerical_limit_probes.json` and `corrected_rng_probes.json`: additional
  checks and the label correction.
- `pairs/`: exact saved graph pairs and their configuration/structural metadata.
- `environment.json`: package versions, thread limits, and Git commit.
- `confirmation_plan.json`: selected budgets.

The reusable sweep script is
`fixed_pair_seed_set_classification/audits/calibrate_faq_goat.py`.
For a fresh calibration with the recommended transport limit, run from the
repository root using the `ds` environment:

```bash
python -m fixed_pair_seed_set_classification.audits.calibrate_faq_goat \
  --output research_notes/sgm_calibration_5000_new \
  --phase confirm \
  --plan research_notes/sgm_calibration_2026-10-13/confirmation_plan.json \
  --sets 12 --threads 4 --transport-max-iter 5000
```

This reproduces the pilot seed sets; main collection should use fresh seed sets.
Existing data collection presets were not changed by this calibration.
