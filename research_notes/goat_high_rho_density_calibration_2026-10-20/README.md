# GOAT calibration at high correlation and mean degrees 10 and 12

Completed 20 October 2026: 504 GOAT runs. The purpose is to keep correlation
high, reducing graph density instead of reducing rho to balance the outcomes.
This supersedes the earlier proposed degree-16 intermediate-rho settings.

## Decision

**ER: use target mean degree 10, rho=0.70, k=3.** On the selected fixed pair,
100 fresh confirmation sets gave 58 successes (58%; approximate 95% Wilson
interval 48.2–67.2%). The adjacent budget k=2 gave 39/100 successes.
Both have useful class variation; k=3 is closer to 50% and inside the
approximate 40–60% target. There is no reason to tune toward exactly 50%.

ER's probability is p=10/599=0.01669449081803005. The saved Graph A and B have
mean degrees 9.82 and 9.85. Keep GOAT regularizer=100 and the solver settings
below. Use a new 1,000-set collection with the existing 400 random train /
200 random test / 200 degree-weighted train / 200 degree-controlled train
design, excluding calibration sets.

**PAPER: no balanced budget found in the tested grid on this pair.** Retaining
rho=0.62, it remained easy at both densities. Do not start a nominally balanced
PAPER classification collection based on these settings. The next diagnostic
should assess a small, predefined group of replication pairs at the same high
rho, reporting all pairs, before changing further parameters. Do not silently
choose a harder pair or lower rho to manufacture balance.

`recommended_settings.json` records the full ER configuration and explicitly
records PAPER as unresolved. No production presets were changed and no main
experiment was launched.

## Results

The initial positive-budget screen used 12 random sets per cell. The singleton
checks used 20 sets; confirmation used 100 sets. Entries below retain these
denominators rather than pooling unlike samples.

| Model | Target mean degree | rho | k=1 | k=2 screen | k=4 | k=8 | k=12 | k=20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ER | 10 | 0.70 | Not tested | 4/12 | 11/12 | 12/12 | 12/12 | 12/12 |
| ER | 12 | 0.70 | 12/20 | 9/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| PAPER | 10 | 0.62 | 18/20 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| PAPER | 12 | 0.62 | 20/20 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |

| Model | Target mean degree | rho | k | Fresh confirmation successes | Approximate 95% interval |
|---|---:|---:|---:|---:|---:|
| ER | 10 | 0.70 | 2 | 39/100 (39%) | 30.0–48.8% |
| ER | 10 | 0.70 | 3 | 58/100 (58%) | 48.2–67.2% |

The one no-seed reference run failed for ER at each density and succeeded for
PAPER at each density. These controls are not repeated-run stability estimates.
At degree 12, the screened ER transition is close to singleton budgets;
there are only 600 distinct singleton sets, so that regime does not support
the intended 1,000-distinct-set design.

PAPER's shared tree is unchanged as ER background density falls. Its measured
overall edge correlation was 0.7055 at degree 10 and 0.6912 at degree 12,
despite the background rho staying at 0.62. The larger proportion of exactly
shared tree structure is a plausible contributor to this pair remaining easy;
no causal ablation was performed.

These are calibration results on one fixed pair per model/density, not
graph-family averages or evidence of improved feature-prediction ROC-AUC.
Confidence intervals describe conditional seed-set sampling uncertainty and
do not include graph-pair variation or adaptive parameter selection.

## Fixed choices

| Model | n | rho | Graph-pair generation seed | GOAT regularizer |
|---|---:|---:|---:|---:|
| ER | 600 | 0.70 | 88424077 | 100 |
| PAPER | 600 | 0.62 on ER background | 7301002 | 300 |

Both target mean degrees, 10 and 12, are tested. ER uses p=d/(n-1).
PAPER uses the existing shared PA tree (alpha=0, beta=1) plus an ER background.
Its background probability is

    p = (d - 2(n-1)/n) / ((n-1) - 2(n-1)/n).

This accounts for ER edges that overlap existing tree edges. The specified
mean degree is an expectation; the realized degrees differ slightly.

Changing density changes Graph A. The existing project generator and original
pair RNG seeds are retained. Within a model and density, one pair is frozen
across budgets. Different densities use common generation random numbers,
so these are coupled comparisons, not independent graph-pair replications.
Both graphs pass connectivity and discrete color refinement; no replacement
draws were needed. A later low-rho comparison at the same density can preserve
Graph A; the script verifies that property without running low-rho GOAT.

GOAT retains the October 13 solver settings: barycenter initialization,
init_perturbation=0, shuffle_input=True, n_init=1, max_iter=100, tol=0.01,
transport_tol=0.05, transport_max_iter=5000, and four BLAS threads.
Success is unseeded accuracy >=0.90.

The initial screen uses 12 uniformly random seed sets for each positive budget
in {2,4,8,12,20}, and one no-seed reference run. The no-seed control is not a
seed-set success-rate estimate. Positive-budget sets are distinct within each
cell and shared across densities for the same model and budget. Different
budgets use separate sampling streams. Confirmation uses a separate stream
and excludes screening sets; main experiment train/test sets must be fresh.
The sampling convention also matches the earlier degree-16 calibration, so
vertex sets may be reused across different graph settings as coupled
comparisons. "Fresh confirmation" means excluded from the screen/refinement
sets in this calibration, not that those vertex IDs were never evaluated on
another graph.

| Model | Target degree | ER/background p | Realized degree A | Realized degree B |
|---|---:|---:|---:|---:|
| ER | 10 | 0.01669449081803005 | 9.82 | 9.85 |
| ER | 12 | 0.02003338898163606 | 11.8067 | 11.8867 |
| PAPER | 10 | 0.013405843630130487 | 10.04 | 10.1067 |
| PAPER | 12 | 0.016755908677226816 | 11.9367 | 11.9733 |

## Validation

All 504 runs converged within 100 outer iterations, had finite soft solutions,
and emitted no numerical warnings. Saved permutations were checked for
bijectivity, preservation of the seed matches, and exact agreement with the
reported unseeded accuracies and success labels. Within-cell uniqueness and
disjointness of confirmation sets from screen/refinement sets were verified.
Both required high rho values were checked in every saved pair configuration.
See `validation.json`.

There were 342 degree-10 runs and 162 degree-12 runs. Total summed solver time
was approximately 11.6 minutes; some screens ran concurrently, so this is not
wall-clock elapsed time.

## Files and reproduction

Each `degree_10/` or `degree_12/` directory contains exact graph pairs and
metadata, the density configuration, environment and source hashes, run-level
outcomes including final permutations, and an aggregate summary. Actual
calibration budgets are in `runs.csv`; `config.n_seeds` in pair-generation
metadata is inherited from the old source configuration and is not used.
`all_summary.csv` combines the two densities with an explicit degree column.
`recommended_settings.json` records the decision; `validation.json` records
the independent checks of saved results.

From the repository root in the existing `ds` environment:

```bash
python -m fixed_pair_seed_set_classification.audits.calibrate_goat_density \
  --mean-degree 10 \
  --output research_notes/goat_high_rho_density_calibration_2026-10-20/degree_10 \
  --budgets 0 2 4 8 12 20 --sets 12 --phase screen

python -m fixed_pair_seed_set_classification.audits.calibrate_goat_density \
  --mean-degree 12 \
  --output research_notes/goat_high_rho_density_calibration_2026-10-20/degree_12 \
  --budgets 0 2 4 8 12 20 --sets 12 --phase screen

python -m fixed_pair_seed_set_classification.audits.calibrate_goat_density \
  --mean-degree 12 \
  --output research_notes/goat_high_rho_density_calibration_2026-10-20/degree_12 \
  --models sparse_er paper_pa --budgets 1 --sets 20 --phase refine

python -m fixed_pair_seed_set_classification.audits.calibrate_goat_density \
  --mean-degree 10 \
  --output research_notes/goat_high_rho_density_calibration_2026-10-20/degree_10 \
  --models paper_pa --budgets 1 --sets 20 --phase refine

python -m fixed_pair_seed_set_classification.audits.calibrate_goat_density \
  --mean-degree 10 \
  --output research_notes/goat_high_rho_density_calibration_2026-10-20/degree_10 \
  --models sparse_er --budgets 2 3 --sets 100 --phase confirm
```

No production presets are changed, and the main datasets are not collected
by these commands.
