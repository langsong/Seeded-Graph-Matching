# GOAT higher-correlation calibration — 20 October 2026

**Historical result, not selected for the next main experiment.** The user
requested retaining ER rho=0.70 and PAPER rho=0.62 and reducing mean degree to
10 or 12 instead. See the subsequent
[density calibration](../goat_high_rho_density_calibration_2026-10-20/README.md).
The outcomes below remain valid; the lower-rho recommendations are superseded.

This directory records a completed fixed-Graph-A seed-budget calibration:
745 main solver runs and one additional numerical-limit check. The practical
recommendation is ER rho=0.60, k=3 and PAPER rho=0.45, k=9.

## Recommended main experiments

| Model | n | Density parameters | rho | k | Mean degree A / B | GOAT regularizer | Fresh confirmation success | Approximate 95% interval |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ER | 600 | p=16/599 | 0.60 | 3 | 15.88 / 16.09 | 100 | 45/100 (45%) | 35.6–54.8% |
| PAPER | 600 | PA alpha=0, beta=1; background p=0.023378 | 0.45 | 9 | 15.83 / 15.69 | 300 | 57/100 (57%) | 47.2–66.3% |

The original ER rho=0.50 and PAPER rho=0.40 collections both used k=12.
The proposed new rhos are higher, while keeping the exact original Graph A
and the same marginal graph-generation parameters. Expected mean degree is
16 for ER and approximately 15.953 for PAPER.

The two confirmation budgets were selected before their confirmation runs:
ER k=3 lies between the screen's k=2 (25%) and k=4 (65%); PAPER k=9
lies between k=8 (40%) and k=12 (90%). Do not tune further to force exactly
50%. Both observed rates satisfy the pre-stated approximate 40–60% target.

Use 1,000 fresh distinct seed sets per model for the main experiment, with
400 random training, 200 random test, 200 degree-weighted training, and
200 degree-controlled training sets. Explicitly exclude calibration sets.
The supplied `recommended_settings.json` records the complete configurations,
exact saved pair paths, solver settings, and confirmation outcomes.

These are calibration estimates on one fixed pair per model, not held-out
model-performance results or graph-family averages. They say nothing yet about
feature-prediction AUC. Changing both rho and k also prevents attributing any
later AUC change to rho alone.

Production presets were not changed. The existing collector CLI has no rho
override; a subsequent collection must explicitly use the recommended
configurations and a new output directory, rather than run the old presets.

## Design

- ER: 600 vertices, p = 16/599, original graph-pair RNG seed 88424077.
- PAPER: 600 vertices, PA alpha = 0, beta = 1, ER background p = 0.023378,
  original graph-pair RNG seed 7301002.
- Each pair is regenerated with the original project generator and RNG seed,
  changing only rho. Assertions check that Graph A and the truth permutation
  are bit-for-bit identical to the saved October 13 pairs.
- Changing rho with the same generation RNG couples the B draws through common
  random numbers. They are valid conditional draws, not independent B
  replications. There is one A and one B per model/rho in this calibration.
- Both graphs must pass the existing connectivity and discrete color-refinement
  checks. No replacement graph draws were needed.
- PAPER's rho refers only to the ER component; the PA tree is shared exactly.
- GOAT settings match the existing collection: one initialization/run per set,
  barycenter initialization, init_perturbation = 0, shuffle_input = True,
  max_iter = 100, tol = 0.01, four BLAS threads, transport_tol = 0.05,
  transport_max_iter = 5000. Regularizer: ER 100, PAPER 300.
- Success means final accuracy on unseeded vertices >= 0.90.
- Every positive-budget screen uses 20 distinct uniformly random seed sets.
  The k=0 control is just one run of the sole empty set, not a success-rate
  estimate over multiple seed sets.
- A confirmation phase uses a separate sampling RNG stream and excludes sets
  used in earlier phases in its output directory, as well as the original
  October 13 GOAT collection. Main experiment train/test sets must be fresh.
- Screen seed sets are shared across rho settings at a given model and k;
  different k values use separate sampling streams, not nested seed sets.
- Confidence intervals in summary.csv use the binomial Wilson approximation.
  They describe seed-set sampling uncertainty for this fixed pair, not
  uncertainty across graph pairs, and do not account for adaptive calibration.

## Initial screen

The originally proposed rho values were too easy for a balanced multi-seed
study on these pairs.

| Model | rho | k=1 | k=2 | k=4 | k=8 | k=12 |
|---|---:|---:|---:|---:|---:|---:|
| ER | 0.70 | 8/20 | 17/20 | 20/20 | 20/20 | 20/20 |
| PAPER | 0.62 | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 |

The no-seed control failed for ER and succeeded for PAPER. A singleton study
on 600 vertices has only 600 distinct seed sets, so it cannot use the existing
1,000-distinct-set collection design.

## Intermediate screens

All entries below use 20 random sets per positive budget. The same sets are
reused across rhos at a given model/k. No feature/outcome model was fitted.

| Model | rho | k=1 | k=2 | k=4 | k=6 | k=8 | k=12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ER | 0.60 | 1/20 | 5/20 | 13/20 | 20/20 | 20/20 | 20/20 |
| PAPER | 0.50 | 18/20 | 16/20 | 19/20 | 19/20 | 20/20 | 20/20 |
| PAPER | 0.45 | 0/20 | 0/20 | 2/20 | Not tested | 8/20 | 18/20 |

PAPER rho=0.50 was still too easy at the screened budgets, motivating the
smaller increase to 0.45. The empty-set reference succeeded at 0.50 and failed
at 0.45; this is one reference outcome per pair, not a stability audit.

## Validation and numerical checks

All 745 primary runs had finite soft solutions and no numerical warnings.
All saved final permutations were independently checked for bijectivity,
preservation of seed matches, and agreement with the reported unseeded accuracy
and success label. Source Graph A and truth were verified against the original
saved files; within-cell seed sets are unique, and both confirmation samples
are disjoint from every screening sample for their model across all rhos.
See `validation.json`.

Three runs reached the outer iteration cap: two in the initial ER rho=0.70
screen and one in the ER rho=0.60 confirmation. The latter was rerun at
max_iter=200 with the same RNG seed: accuracy changed from 0.03518 to
0.03685, converged, and remained a failure. No confirmation label changed.
See `intermediate/confirmation_numerical_checks.json`. The two initial-screen
cap cases were not rerun; no recommended setting depends on their labels.
All 100 PAPER confirmation runs converged.

The main runs consumed approximately 19.8 summed solver minutes; some model
runs were concurrent, so this is not wall-clock elapsed time.

## Output layout

- Root: initial screen at ER rho=0.70 and PAPER rho=0.62.
- `intermediate/`: screens at ER rho=0.60 and PAPER rho=0.50, and the ER
  k=3 confirmation.
- `recommended_rhos/`: PAPER screen at rho=0.45 and subsequent checks.
- Each directory contains run-level outcomes and permutations in `runs.csv`,
  aggregate results in `summary.csv`, exact graph pairs and metadata in
  `pairs/`, and an environment record with source hashes.
- `rho_overrides.json` records alternative rho values and the wrapper hash.
- `recommended_settings.json`: complete proposed collection settings.
- `validation.json`: checks against saved graph arrays and final matchings.

## Reproduction

Run from the repository root in the existing `ds` environment. The commands
resume completed cells rather than overwrite them. Use new output directories
to run from scratch. These scripts calibrate only; they do not change the
production presets or start the main 1,000-set collections.

```bash
python -m fixed_pair_seed_set_classification.audits.calibrate_goat_high_rho \
  --output research_notes/goat_high_rho_calibration_2026-10-20 \
  --budgets 0 1 2 4 8 12 --sets 20 --phase screen

python -m fixed_pair_seed_set_classification.audits.calibrate_goat_rho_override \
  --er-rho 0.60 --paper-rho 0.50 \
  --output research_notes/goat_high_rho_calibration_2026-10-20/intermediate \
  --budgets 0 1 2 4 6 8 12 --sets 20 --phase screen

python -m fixed_pair_seed_set_classification.audits.calibrate_goat_rho_override \
  --er-rho 0.60 --paper-rho 0.50 \
  --output research_notes/goat_high_rho_calibration_2026-10-20/intermediate \
  --models sparse_er --budgets 3 --sets 100 --phase confirm

python -m fixed_pair_seed_set_classification.audits.calibrate_goat_rho_override \
  --er-rho 0.60 --paper-rho 0.45 \
  --output research_notes/goat_high_rho_calibration_2026-10-20/recommended_rhos \
  --models paper_pa --budgets 0 1 2 4 8 12 --sets 20 --phase screen

python -m fixed_pair_seed_set_classification.audits.calibrate_goat_rho_override \
  --er-rho 0.60 --paper-rho 0.45 \
  --output research_notes/goat_high_rho_calibration_2026-10-20/recommended_rhos \
  --models paper_pa --budgets 9 --sets 100 --phase confirm
```
