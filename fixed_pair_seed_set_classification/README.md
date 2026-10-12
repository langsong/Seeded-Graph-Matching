# Fixed-pair seed-set selection

Each benchmark keeps one correlated graph pair fixed, varies the seed set in Graph A, and measures SGM performance. The current focus is sparse ER, true PAPER (`paper_pa`), and fixed-weight Pareto–Chung–Lu (`chung_lu`). The older `paper` name refers to the historical random-weight Chung–Lu generator.

| Model | Pair settings | Seed count |
| --- | --- | ---: |
| `sparse_er` | 600 vertices, edge probability 0.010, correlation 0.70 | 26 |
| `paper_pa` | 600 vertices, attachment alpha=0 and beta=1, ER background probability 0.005 and correlation 0.62 | 13 |
| `chung_lu` | 600 vertices, fixed Pareto quantile weights with alpha=2, weight scale 1.9260087733615983, no ER floor, correlation 0.70 | 17 |

These budgets were calibrated so that a uniformly random seed set has roughly a 50% chance that the **highest-objective result among five SGM runs** reaches 0.90 unseeded accuracy. The calibration is approximate and varies across graph pairs; every individual run is still saved for grouped-binomial analysis.

## Seed sets

By default each pair gets 500 unique seed sets, chosen from Graph A only (no SGM outcome is used):

| Group | Count | Purpose |
| --- | ---: | --- |
| Uniform random training | 200 | Typical sets for fitting |
| Uniform random test | 100 | Held-out reference distribution |
| Degree-weighted training | 100 | About 16–17 sets at each exponent −1, −0.5, 0.5, 1, 2, and 3 (weights `max(degree, 1) ** gamma`) |
| Degree-controlled training | 100 | 50 pairs: a low and a high value of one feature at equal seed-to-nonseed degree |

Swap-path sets are no longer part of the default 500, but can still be added with `--swap-paths`. Their starting rules are top degree, top harmonic centrality, greedy one-hop coverage, greedy witness-pair mass, greedy one-hop signature separation, and farthest-point spread within the largest component; ties are broken at random. Swap counts are 0, 2, 5, 10, and half the budget.

**Contrasts (sampler version 2, 6 Oct 2026).** Pairs come from a pool of 4,000 sets drawn with exponents −1 to 3 (including uniform), restricted to the 10th–90th percentile of the random sets' seed-to-nonseed degree (where the calibrated budget puts success near 50%) and split into 6 equal-count bins. Within a bin, the two sets of a pair have exactly the same number of seed-to-nonseed edges and share a `contrast_pair_id`. Options: `--contrast-features` (repeat a name to give it more pairs, e.g. `unique_signatures_r3 unique_signatures_r3 pair_mass`), `--contrast-bins`, `--contrast-percentiles`; the range and bin edges are saved in the metadata. Version 1 (the `sgm_pilot_400` pilot) used ten bins over the whole pool, matched on total seed degree and had no pair IDs. Random and degree-weighted sets are identical in both versions for the same `--random-seed`.

## Features and outcomes

The panel (version 1) has nine Graph-A features: mean seed-to-nonseed degree, mean seed harmonic centrality, one-hop coverage, witness overlap, witness-pair mass, mean nearest-seed distance, mean number of seeds within three hops, distance-signature uniqueness, and mean seed-neighbourhood Jaccard overlap. Distance-signature uniqueness (`..._r3`) is the fraction of nonseeds whose vector of distances to the seeds, capped at 3, is unique: in words, whose 2-hop seed profile (which seeds are neighbours, which are neighbours' neighbours) no other vertex shares. It is a truncated version of the resolving-set (metric dimension) idea. New collections also store caps 2 and 4 (`..._r2`, `..._r4`) as robustness checks, and the metadata records the cap `floor(log n / log mean degree)` (3 for these pairs).

Every set gets the same five solver RNG seeds, **2026–2030**, at `max_iter=100`. A run succeeds at final unseeded accuracy ≥ 0.90. The main CSV has one row per set (features, mean accuracy and H0, number and fraction of successful runs, best-objective accuracy and success); `_runs.csv` has every run.

## Analysis notebook

`seed_set_classification.ipynb` (helpers in `count_analysis.py`) rebuilds outcomes from the run-level CSV and models the number of successful runs with a regularized grouped-binomial logistic model of the single-run success probability p(S). It compares degree only, signatures only, both, and the full panel. C and the primary model are chosen by repeated 5-fold cross-validation on the random training sets (constructed training sets stay in every fit unless `TRAIN_ON = 'random'`); the 100 random test sets are used once, with the random-training success rate as the no-feature reference. Best-objective-of-R success (ties to the earliest run), any-run success and objective-selection misses come directly from the runs; `1 - (1 - p) ** R` is shown only as a diagnostic and tends to overstate restart success when predictions are imprecise. The notebook also reports solver noise, feature correlations with and without controlling for degree, residuals of constructed groups, and contrast pairs.

## Running a collection

From the repository root, activate the project environment and collect the three graph pairs for each model. Each command below writes 500 seed sets and 2,500 SGM runs per pair into a separate subfolder of `data/sgm_balanced_2026-10-07/`:

```bash
conda activate ds

python -m fixed_pair_seed_set_classification.candidate_data \
  --graph-model sparse_er \
  --pair-seeds 4192889773 18472911 37051993

python -m fixed_pair_seed_set_classification.candidate_data \
  --graph-model paper_pa \
  --pair-seeds 424242 18472911 37051993

python -m fixed_pair_seed_set_classification.candidate_data \
  --graph-model chung_lu \
  --pair-seeds 3165569293 18472911 37051993
```

The defaults already encode 200 uniform training sets, 100 uniform test sets, 100 degree-weighted sets, 100 degree-controlled contrast sets, and five SGM runs per set. The collection limits BLAS/OpenMP threads to 4 (`--blas-threads`, matching the calibration; thread count changes SGM outcomes). The metadata file is written at the start with `"status": "running"`, updated every 25 sets and marked `"complete"` at the end; it records library versions, thread settings and the git commit. Both CSVs are flushed after every set, so an interrupted run keeps its finished rows (the notebook only reads complete collections). Graph pairs are cached inside each pair folder and checked against the generating settings. Existing outputs are protected; use a new `--output-dir`, or add `--overwrite` only when intentionally restarting a collection. At 500 sets × 5 runs, expect roughly 65–100 minutes per graph pair.

To analyse a result, set `GRAPH_MODEL` and `DATA_DIR` in the first code cell of `seed_set_classification.ipynb` to the model and one pair folder, then run the notebook from top to bottom. For example, the checked-in default points to the first ER pair. Do not combine graph pairs into one folder; pair-specific evaluation and later leave-one-pair-out analysis need their identities preserved.

To check that a saved collection reproduces (features and every SGM run of a few sets), use the same thread setting:

```bash
python -m fixed_pair_seed_set_classification.audits.reproduce_saved_sets \
  --graph-model sparse_er \
  --data-dir fixed_pair_seed_set_classification/data/sgm_balanced_2026-10-07/sparse_er/pair_1_seed_4192889773
```

Calibration evidence is in `audits/sgm_settings_calibration_2026-09-30.md`; next steps are in `research_notes/2026-10-06_next_steps.md`. `--graph-model all` includes the historical models. The Enron benchmark reads `utils/enron.mat` and keeps the 129 vertices active in both weekly snapshots.

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
