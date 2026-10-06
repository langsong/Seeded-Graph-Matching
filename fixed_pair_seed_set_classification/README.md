# Fixed-pair seed-set selection

Each benchmark keeps one correlated graph pair fixed, varies the seed set in Graph A, and measures SGM performance. The current focus is sparse ER, true PAPER (`paper_pa`), and fixed-weight Pareto–Chung–Lu (`chung_lu`). The older `paper` name refers to the historical random-weight Chung–Lu generator.

| Model | Pair settings | Seed count |
| --- | --- | ---: |
| `sparse_er` | 600 vertices, edge probability 0.010, correlation 0.70 | 30 |
| `paper_pa` | 600 vertices, attachment alpha=0 and beta=1, ER background probability 0.005 and correlation 0.62 | 16 |
| `chung_lu` | 600 vertices, fixed Pareto quantile weights with alpha=2, ER floor 0.005, correlation 0.70 | 22 |

For each model, the default pilot draws 400 unique seed sets:

| Group | Count | Purpose |
| --- | ---: | --- |
| Uniform random training | 80 | Typical sets for fitting |
| Uniform random test | 40 | Held-out reference distribution |
| Degree-weighted training | 160 | About 26–27 sets at each exponent −1, −0.5, 0.5, 1, 2, and 3 |
| Swap-path training | 80 | Rule-based starting sets and partial replacements with random nonseeds |
| Degree-controlled training | 40 | High and low tails of coverage, witness-pair mass, and three-hop distance-signature uniqueness within mean-degree bins |

The six swap starting rules are top degree, top harmonic centrality, greedy one-hop coverage, greedy witness-pair mass, greedy one-hop adjacency-signature separation, and farthest-point spread within the largest connected component. Swap counts are 0, 2, 5, 10, and half the seed budget where distinct. Degree weights are proportional to `max(degree, 1) ** gamma`, which also defines negative exponents for isolated vertices. The degree-controlled sets are selected from a pool of 4,000 degree-weighted sets using Graph-A features only. The pool is split into ten equally sized mean-degree bins; within each bin, high and low feature sets are paired at the same total seed degree whenever possible. No SGM outcome is used to choose candidates. The CSV records each set's sampling strategy, starting rule, swap count, weighting exponent, contrast, and degree bin.

The fixed-pair feature panel is frozen at version 1 with nine Graph-A predictors: mean seed-to-nonseed degree, mean seed harmonic centrality, one-hop nonseed coverage, witness overlap, witness-pair mass, mean nearest-seed distance among reachable nonseeds, mean number of seeds within three hops, the fraction of nonseeds with unique seed-distance signatures capped at three hops, and mean pairwise seed-neighborhood Jaccard overlap. These are computed for every candidate without inspecting Graph B or SGM outcomes.

Every set is evaluated with the same five solver RNG seeds, **2026–2030**, at `max_iter=100`. A run succeeds when final unseeded accuracy is at least 0.90. The main CSV has one row per set with mean H0, mean final accuracy, success fraction, and best-by-objective accuracy. The companion `_runs.csv` has all five individual outcomes. The notebook uses the random test rows for its held-out evaluation.

Run from the repository root in the environment containing the project dependencies, one model at a time:

```bash
conda activate ds
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m fixed_pair_seed_set_classification.candidate_data --graph-model sparse_er
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m fixed_pair_seed_set_classification.candidate_data --graph-model paper_pa
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m fixed_pair_seed_set_classification.candidate_data --graph-model chung_lu
```

Outputs go to `data/sgm_pilot_400/`. Its graph-pair files live in `graph_pairs/`, leaving both the historical datasets in `data/` and any earlier `data/sgm_multi_rng/` collection intact. A cached pair is checked against the graph-generating settings before reuse; to change those settings later, use a new `--output-dir`. Existing output CSVs are protected unless `--overwrite` is passed. To change the collection size, set the five group counts with `--random-train`, `--random-test`, `--degree-weighted`, `--swap-paths`, and `--decorrelated`; `--pool-size` changes the contrast-selection pool. In the notebook, use `DATA_FOLDER = "sgm_pilot_400"` and choose `GRAPH_MODEL` at the top, then run the cells in order.

At 400 sets and five runs each, the calibration timings imply roughly 50–80 minutes of SGM work per model, plus candidate generation and feature extraction. Keep the thread environment consistent across models because it affected ER and PAPER solver outcomes in the calibration. With only 40 random test sets, treat the notebook's AUC and top-pick estimates as preliminary.

The calibration evidence is in `audits/sgm_settings_calibration_2026-09-30.md`. The other audit scripts and historical model definitions remain available. `--graph-model all` includes those models and can take considerably longer. The Enron benchmark reads `utils/enron.mat` and retains the 129 vertices active in both weekly snapshots; its vertex correspondence is used only to evaluate matching, not to construct seed-set predictors.
