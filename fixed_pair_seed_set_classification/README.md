# Fixed-pair seed-set selection

The current SGM collection compares standard FAQ-SGM and GOAT-SGM on one fixed
primary pair per model. Each solver receives **1,000 distinct seed sets and one
SGM run per set**. Graph pairs are shared between the solvers; their calibrated
seed budgets differ. The `sparse_er` generator name is retained for ER even
though the new family has mean degree about 16.

## SGM settings

| Model | Graph parameters | rho | Pair RNG seed | FAQ seeds | GOAT seeds | GOAT regularizer |
|---|---|---:|---:|---:|---:|---:|
| `sparse_er` | n=600, p=16/599 (0.0267111853) | 0.50 | 88424077 | 29 | 12 | 100 |
| `paper_pa` | n=600, alpha=0, beta=1, background p=0.023378 | 0.40 | 7301002 | 31 | 12 | 300 |
| `chung_lu` | n=600, Pareto alpha=2, weight scale=5.452064, ER floor=0 | 0.45 | 18472911 | 27 | 9 | 300 |

Common solver settings are `n_init=1`, barycenter initialization,
`init_perturbation=0`, `shuffle_input=True`, `max_iter=100`, `tol=0.01`, and four
BLAS threads. GOAT uses `transport=True`, `transport_tol=0.05`, and
`transport_max_iter=5000`. PAPER's rho applies to the ER background; its PA tree
is shared by the two graphs.

Both graphs must be connected and have a discrete color refinement, which
certifies the absence of nontrivial automorphisms. A nondiscrete refinement is
inconclusive and is rejected by this collection. Supplied pair seeds are not
silently replaced. The calibrated primary seeds above passed these checks.

The seed budgets were selected for approximately balanced **single-run random
seed-set outcomes**, not best-of-five outcomes. The chosen fixed pairs gave
FAQ/GOAT success rates of 8/12 and 7/12 (ER), 7/12 and 5/12 (PAPER), and 4/12 and
7/12 (Chung-Lu) in calibration. Larger collections can differ. Evidence and
repeat-stability results are in
`research_notes/sgm_calibration_2026-10-13/README.md`. FAQ remains especially
variable across RNG seeds on Chung-Lu; a one-run label is one observed outcome.

## Seed sets

The following allocation applies to each solver and pair:

| Group | Count | Purpose |
|---|---:|---|
| Uniform random training | 400 | Fit on typical seed sets |
| Uniform random test | 200 | Hold out for evaluation |
| Degree-weighted training | 200 | Explore a wider feature range |
| Degree-controlled training | 200 | 100 contrast pairs for effects beyond degree |
| **Total** | **1,000** | **1,000 single-initialization SGM runs** |

All selection uses Graph A and no SGM outcomes. Seed sets are unique across
training/test and sampling groups within each dataset. The sampling seed is
1234 by default. Degree weighting cycles through exponents -1, -0.5, 0.5, 1,
2, and 3 with weights `max(degree, 1) ** gamma`. Contrasts use a 4,000-set pool,
the 10th-90th percentile of random sets' seed-to-nonseed degree, and six bins.
Contrast sets retain their low/high feature labels and `contrast_pair_id`.
The shared sampler and percolation's sampling defaults are unchanged.

Both solvers use the same sampling design and graph pair. Because their k
values differ, these balanced collections compare predictors for two solver
regimes. They are not a solver comparison at an equal seed budget. For a common
budget comparison, supply `--n-seeds K --solver both` and a new output directory;
the sampled seed sets are then identical for FAQ and GOAT.

## Features and labels

The Graph-A predictor panel is unchanged: seed-to-nonseed degree, harmonic
centrality, coverage, witness overlap, witness-pair mass, nearest-seed distance,
three-hop seed count, capped distance-signature uniqueness, and neighborhood
Jaccard overlap. Signature caps 2 and 4 are also recorded. The primary signature
cap remains 3: distances 1 and 2 are distinct, and all distances at least 3 share
the capped category. The formula `floor(log(n)/log(mean degree))` is recorded in
metadata and is normally 2 in this denser family.

A set's binary label is **final unseeded accuracy >= 0.90 in its single run**.
`sgm_success_count` and `sgm_success_rate` are therefore 0 or 1. Historical
majority/best-objective columns remain for file compatibility; with one run
their labels and accuracies agree with that run. Each candidate gets an
independent reproducible solver RNG seed, saved in the run CSV. `sgm_seed` in
metadata is the base for this policy, not the seed used for every candidate.

GOAT's `h0_accuracy` uses a deterministic one-to-one assignment projection of
the first soft direction. Additional diagnostic columns store row-argmax
accuracy, mass on the true matches, row entropy, solver warnings, and numerical
validity. These solver/ground-truth diagnostics are not predictor features.
Metadata stores the solver options, RNG policy, graph-pair checks, warning
counts, convergence counts, versions, and thread settings.

## Collect the primary pairs

Run from the repository root after the diagnostic notebook finishes:

```bash
conda activate ds
cd /Users/langsong/Desktop/Seeded-Graph-Matching

python -m fixed_pair_seed_set_classification.candidate_data \
  --graph-model focus --solver both
```

This collects one primary pair for each of three models and both solvers:
**six datasets, 6,000 seed sets, and 6,000 SGM runs**. It does not create three
pairs per model. To collect one model at a time instead:

```bash
python -m fixed_pair_seed_set_classification.candidate_data --graph-model sparse_er --solver both
python -m fixed_pair_seed_set_classification.candidate_data --graph-model paper_pa --solver both
python -m fixed_pair_seed_set_classification.candidate_data --graph-model chung_lu --solver both
```

Use either the combined command or the three individual commands. Replace
`both` with `faq` or `goat` to collect one solver. No count arguments are needed
for the default 1,000-set design. `--n-seeds`, `--transport-regularizer`,
`--transport-tol`, `--transport-max-iter`, and `--blas-threads` are optional
overrides. A custom `--n-seeds` overrides the budget of each selected solver.

## Output locations

The default root is
`fixed_pair_seed_set_classification/data/sgm_fixed_pairs_2026-10-13/`:

```text
sgm_fixed_pairs_2026-10-13/
  graph_pairs/<model>/pair_seed_<seed>/       # one pair shared by both solvers
  faq/<model>/pair_1_seed_<seed>/             # FAQ outcome files
  goat/<model>/pair_1_seed_<seed>/            # GOAT outcome files
```

For example, ER FAQ writes to
`.../faq/sparse_er/pair_1_seed_88424077/`; ER GOAT writes to
`.../goat/sparse_er/pair_1_seed_88424077/`.
Each dataset has `<model>_seed_sets.csv` (one row per set),
`<model>_seed_sets_runs.csv` (one row per algorithm run), and
`<model>_seed_sets.metadata.json`. A completed default dataset has
`status="complete"`, `rows=1000`, `rows_completed=1000`, `solver_runs=1`,
`run_rows=1000`, and `n_init=1`.

Progress is printed every 25 sets. CSVs are flushed after each set and metadata
is updated every 25 sets. Existing outcome files are protected. The collector
does not resume a partial dataset automatically: use `--overwrite` to restart
that solver, or choose a new `--output-dir`. If FAQ completed and GOAT was
interrupted, target only `--solver goat --overwrite` to restart GOAT.
Review numerical warnings and nonconvergence counts printed at completion and
saved in metadata; flagged runs are retained for inspection.

Pilot solver times were approximately one second per FAQ run and two seconds
per GOAT run. Allow roughly 20-30 minutes per FAQ dataset and 35-60 minutes per
GOAT dataset, plus sampling/features; runtime depends on the machine and pair.
Older SGM and percolation data remain in their original directories.

## Analyse or reproduce a collection

Run `seed_set_classification.ipynb` from top to bottom to analyse the saved
one-run collections. Set `SOLVER` to `faq` or `goat` and `GRAPH_MODEL` to
`sparse_er`, `paper_pa`, or `chung_lu` in its first code cell. The notebook
shows the six collection label distributions, then analyses the selected one
with five-fold training cross-validation, held-out ROC-AUC and other
classification metrics, feature importance, seed-set ranking, and
degree-controlled contrasts. Its models train on uniformly random seed sets;
the 200 random test sets remain held out, and constructed sets are analysed
separately. A single observed label does not directly measure a seed set's
success probability across solver RNG seeds.

For a reproduction check using the recorded solver settings and shared pair:

```bash
python -m fixed_pair_seed_set_classification.audits.reproduce_saved_sets \
  --graph-model sparse_er \
  --data-dir fixed_pair_seed_set_classification/data/sgm_fixed_pairs_2026-10-13/goat/sparse_er/pair_1_seed_88424077
```

For optional additional graph pairs, supply `--pair-seeds SEED1 SEED2 ...` for
one model. They use the same solver budgets, are checked structurally, and are
written to separate numbered folders. Analyse pairs separately. For repeat
audits, `--solver-runs 5` explicitly requests five `n_init=1` runs; use a new
output directory. The old auxiliary graph models remain accessible with
`--graph-model all --solver faq`; GOAT presets cover only the three focus models.


## Fixed-pair percolation collection

Percolation uses separately calibrated, denser graph settings. The initial diagnostic uses one fixed reference pair per model. That pair gets 1,000 seed sets and one tie-break run per set:

| Group | Count |
| --- | ---: |
| Uniform random training | 400 |
| Uniform random test | 200 |
| Degree-weighted training | 200 |
| Degree-controlled training | 200 |

The algorithm is `ExpandWhenStuck` with witness threshold `r=2`. ER uses p=0.020, rho=0.70, and k=8; PAPER uses a shared PA tree plus ER background p=0.0167, rho=0.62, and k=7; Chung–Lu uses alpha=2, weight scale 4.076194702720809, no ER floor, rho=0.70, and k=5.

Run the reference pair for each model from the repository root:

```bash
conda activate ds

python -m fixed_pair_seed_set_classification.percolation_data \
  --graph-model sparse_er \
  --pair-seeds 4192889773

python -m fixed_pair_seed_set_classification.percolation_data \
  --graph-model paper_pa \
  --pair-seeds 424242

python -m fixed_pair_seed_set_classification.percolation_data \
  --graph-model chung_lu \
  --pair-seeds 3165569293
```

Outputs go to `data/percolation_fixed_pairs_2026-10-07/<model>/<pair>/`. A completed metadata file has `"status": "complete"`, `"rows": 1000`, `"rows_completed": 1000`, and `"runs_per_seed_set": 1`. Existing outputs are protected; add `--overwrite` only when intentionally restarting a pair.

Run `percolation_seed_set_classification.ipynb` from top to bottom to analyse one collected pair. Change `GRAPH_MODEL` in its first code cell to switch among ER, PAPER, and Chung–Lu. The notebook keeps the 200 random test sets held out for final classification metrics and feature importance.

After the analysis is working on the reference pair, the same fixed-pair analysis can be replicated on additional graph draws by appending `18472911 37051993` to `--pair-seeds`. Those pair datasets should be analysed separately rather than pooled automatically.

## Many pairs for pooled models (30 Oct 2026)

`multi_pair_collection.py` collects seed sets on many fresh pairs per setting, so that one pooled model per setting can be fitted in `multi_pair_models.ipynb`. The pooled weighted score is what best of N uses to screen random seed sets.

| | ER | PAPER | Chung–Lu | Pairs per model |
| --- | ---: | ---: | ---: | ---: |
| Percolation | 8 | 7 | 5 | 50 |
| GOAT | 15 | 12 | 9 | 32 |
| FAQ | 29 | 29 | 27 | 24 |

- **Pairs.** Fresh, structure-checked pairs with round 1's graph settings (`selection_rule_comparison.settings`), from a new seed stream. Round-1, borderline and fixed-pair seeds are never reused, so those pairs stay free for testing. FAQ and GOAT share pairs.
- **Budgets.** Where a single run from a uniform random set succeeded about half the time on round 1's pairs. They match the fixed pairs above except GOAT ER (12 there) and FAQ PAPER (31 there).
- **Seed sets.** 500 per pair, one run each, with round 1's solver settings. All are uniform random sets: 300 random; 100 tail sets, the top 5% of a 20,000-set random pool in degree, pair mass, uniqueness or the sum of their z-scores; and 100 contrast sets, 50 pairs with the same number of seed-to-nonseed edges but low against high pair mass or uniqueness. Tail and contrast sets are chosen by graph-A features only, never by outcomes.

Run from the repository root. It takes about 35–40 hours with three pairs at a time, rows are written set by set, Ctrl-C stops it, and rerunning resumes:

```bash
conda activate ds
python -m fixed_pair_seed_set_classification.multi_pair_collection
python -m fixed_pair_seed_set_classification.multi_pair_collection --status
```

`--sets-per-pair 1000 --pairs percolation=25 goat=16 faq=12` gives 1,000 sets per pair on half as many pairs, in about the same time; choose before starting. Pairs run in order of pair index across all nine settings, so stopping early leaves every setting with a similar number of pairs. Outputs go to `data/multi_pair_2026-10-30/<algorithm>/<model>/pair_XXX/` (`pair.json`, `sets.csv`, `pool_features.npz`; about 0.4 MB per pair). Analyse with `multi_pair_models.ipynb` and `RUN_COLLECTION = False`.

## SBM pairs for SBM's own weights (2 Nov 2026)

`sbm_collection.py` collects the same kind of data as `multi_pair_collection.py`, but on SBM pairs, so SBM gets its own pooled weights. `sbm_models.ipynb` fits them and compares them with the ER, PAPER and Chung–Lu weights.

- **Graphs.** The two main settings of `selection_rule_comparison/sbm_rules.py`: plain SBM ("ER plus communities") and degree-corrected SBM ("Chung–Lu plus communities"). Both have 4 equal blocks of 150 vertices, with 80% of edges inside blocks. The pairs come from their own seed stream.
- **Budgets.** k50 from `sbm_rules`' budget calibration: the budget where a random set succeeds about half the time. A missing calibration runs first and is saved with `sbm_rules`' results.
- **Pairs per setting.** Percolation 30, GOAT 20, FAQ 16.
- **Seed sets and runs.** As in the multi-pair collection: 500 per pair (300 random, 100 tail, 100 contrast), one run each with round 1's solver settings. Every set, and every set of the 20,000-set pool, also records the largest block's share of its seeds and the number of blocks it covers. These come from the true block labels, so they are diagnostics; the saved weights use the four core features only.
- **Direct test.** Section 7 of `sbm_models.ipynb` runs best of 100 and best of 1,000 with the fitted weights on `selection_rule_comparison/sbm_rules`' 20 test pairs per setting, which were never used for fitting. It compares them with random, top degree, degree alone and the averaged weights (about 25 minutes). Section 8 checks whether the score can pick among round 1's rules.

Run from the repository root. It takes about 11 hours with three pairs at a time. Rows are written set by set, Ctrl-C stops it, and rerunning resumes:

```bash
conda activate ds
python -m fixed_pair_seed_set_classification.sbm_collection
python -m fixed_pair_seed_set_classification.sbm_collection --status
```

`--pairs goat=10 faq=8` sets fewer pairs; pairs run in order of pair index, so stopping early leaves every setting with a similar number. Outputs go to `data/sbm_multi_pair/<algorithm>/<setting>/pair_XXX/` (`pair.json`, `sets.csv` and `pool_features.npz`, as in the multi-pair collection). Analyse with `sbm_models.ipynb` and `RUN_COLLECTION = False`. Its last section saves SBM's weights to `data/sbm_multi_pair/pooled_weights.json`.
