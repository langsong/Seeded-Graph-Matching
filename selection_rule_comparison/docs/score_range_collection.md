# Overnight score-range collection

Run from the repository root on the Mac, using the `ds` environment:

```bash
conda activate ds
caffeinate -i python -u -m selection_rule_comparison.collect_score_ranges --hours 8 --jobs 3
```

`caffeinate` keeps the Mac awake while the command runs. Keep it connected to power.
The eight-hour limit includes candidate generation. Workers check the limit
between evaluations and candidate-generation steps; an evaluation already in
progress finishes before the worker stops. The process can finish earlier if all
work is complete. Closing the terminal stops a foreground collection.

Run the same command again to resume. The time allowance starts afresh on each
invocation. Successful evaluations are skipped. Ctrl-C also preserves successful
rows. No plotting or model fitting happens in this command.

Progress is printed and appended to
`selection_rule_comparison/data/score_range_collection/collection.log`.
From a second terminal, or tomorrow:

```bash
python -m selection_rule_comparison.collect_score_ranges --status
```

## Cases and queue

| Cases | Budgets | Unique sets per bank |
| --- | --- | ---: |
| Enron 130–131, 130–132, 145–147, 160–162 | 5, 10, 20 | 150 |
| Worm full and common LCC | 50, 100 | 150 |
| Worm male | 20, 40 | 150 |
| Wikipedia EN/FR | 10, 20 | 100 |
| The 27 learned-selector cells | Each cell's original calibrated budget | 150 |

The queue completes these stages in order:

1. Small real pairs: the same bank for percolation, FAQ and GOAT; one percolation
   run and three solver runs for FAQ and GOAT, on the native B.
2. Synthetic cells: one run per set on the original B of saved fresh pair 0.
3. Wikipedia: one run per algorithm per set on the native B; at most two workers.
4. A second synthetic B, shared by all sets in each cell.
5. A third synthetic B, shared by all sets in each cell.
6. A second solver run for every Wikipedia FAQ/GOAT set.

The full queue is 47 seed banks and up to 32,050 matching evaluations. Finishing
the full queue in eight hours is not guaranteed. The saved data identify exactly
which complete blocks are available. Compare sets at the same draw/repeat budget;
do not silently average different numbers of runs after an interruption.

The synthetic cells comprise nine cells per algorithm, using that algorithm's
calibrated k. They do not form an 81-cell cross product. Graph A is from the first
saved fresh pair, not the graph used to fit the selector. The other nine saved
fresh pairs remain available for subsequent experiments. The worm full/LCC cases
overlap; Enron pairs from one time window are also dependent.

## Candidate construction

All candidate construction uses A, the eligible seed vertices and saved scores.
It never uses matching accuracy, held-out truth or B's edge structure to rank
seeds. Test vertices cannot be seeds.

- Start from a fixed uniform random pool and retain 30 uniform controls (20 for
  Wikipedia).
- Include all six hand rules, exact top-three additive degree/harmonic sets,
  pool-screened high scores, and high values found by sampled single swaps.
  Nonadditive searches do not claim global maxima.
- Follow multiple independent random-to-endpoint swap paths, with up to 24
  positions along each path. This supplies intermediate feature/score values.
- Reserve maxima of the scoring objectives and approximately degree-matched
  low/high pair-mass, coverage and uniqueness contrasts.
- Fill the remaining bank with distinct sets from eight equal-width score
  intervals over the constructed bank. Balance intervals across frozen scores
  and saved predictions, and favour differing seed memberships within intervals.
  Constants or unattainable intervals are reported, not artificially filled.

The categories can overlap; `memberships.csv` retains these memberships. Bank sizes
are capped after deduplication, so a tiny graph with too few possible sets could
produce fewer sets. Random controls retain their primary `random` label even if
they also appear as a contrast or endpoint.

On real pairs, raw features are normalized against one shared 1,000-set uniform
reference pool per pair/budget. Each algorithm uses its existing uniqueness cap
(FAQ 2; percolation/GOAT 3). Each algorithm's three frozen scores and the mean of
compatible synthetic predictors help cover score ranges. Every individual source
prediction and its range flag is saved too; averaging the predictors is a sampling
device and baseline, not a claim that the ensemble transfers well.

On synthetic cells, normalization reproduces the original landscape pool's RNG,
pool size and feature definitions for that fresh pair. The cell's own frozen score
and final selector define its score intervals. No selector is refitted. Synthetic
GOAT keeps the source graph family's regularizer; real GOAT uses 300 throughout.

Every candidate in an algorithm/draw/repeat block receives the same solver seed.
The underlying B and test vertices are fixed within a block. Extra synthetic Bs
follow the existing conditional-B generator and graph checks. Native real pairs
are not redrawn. Percolation's algorithm settings are unchanged.

## Saved data

The new output root is
`selection_rule_comparison/data/score_range_collection/`.
Original experiment files and fitted selectors are not modified.

| File | Contents |
| --- | --- |
| `settings.json` | Frozen sampling/solver design, case list, source-model and code hashes |
| `selector_sources.json`, `environment.json` | Complete saved model snapshots and runtime provenance |
| `<bank>/pair.npz` | A, original shuffled B, true permutation, held-out test vertices |
| `<bank>/pair.json` | Graph hashes, test vertices, normalization, sampling seed and set-bank hash |
| `<bank>/sets.csv` | Unique memberships, primary family, raw/z features and scores |
| `<bank>/memberships.csv` | Rules, endpoints, interval picks and contrast memberships |
| `<bank>/coverage.csv` | Attainable interval bounds and available/selected/control counts |
| `<bank>/predictions.csv` | Each source selector's prediction and extrapolation flag |
| `<bank>/{percolation,faq,goat}_runs.csv` | One successful evaluation per candidate/draw/repeat |
| `<bank>/draw_XX.json` | Additional synthetic B hashes and rejection counts |
| `<bank>/*_errors.csv`, `collection_error.txt` | Failures, if any; failed evaluations are retried on resume |

`sets.csv` is written last as the bank's commit marker. CSV outcome rows are
flushed after every evaluation. An interrupted final row is removed when the
collector resumes. A lock prevents two collectors from writing the same output
root. Changed models, code or sampling settings require a new output root.
Operational changes such as time allowance, worker count or `--only` filters can
resume the same design.

Additional features include coverage by two/three seeds, witness-count quantiles,
internal seed edges, total vertex degree, adjacency-signature collisions and
signature entropy. These can be analyzed tomorrow without rerunning matching.
`predicted_mean` is an accuracy prediction; frozen scores have no probability
interpretation. In synthetic banks, unprefixed `uniqueness` uses that cell's cap;
in shared real banks it uses cap 3. Algorithm-prefixed features are explicit.

For notebook analysis, merge each run file with its bank's `sets.csv` on
`candidate_id`, and retain the graph/budget/algorithm/draw/repeat identifiers.
Use `memberships.csv` for overlapping strategy groups rather than treating them as
additional independent evaluations. `first_B` in the status table counts completed
draw-0/repeat-0 outcomes; `complete` includes optional queue stages.

## Smaller checks and optional controls

```bash
# Small isolated end-to-end collection; never writes the main output folder.
python -m selection_rule_comparison.collect_score_ranges --smoke --hours 0.1

# Only real or only synthetic work, retaining the same frozen design.
python -m selection_rule_comparison.collect_score_ranges --only real --hours 8
python -m selection_rule_comparison.collect_score_ranges --only synthetic --hours 8

# Candidate generation and predictions only, with no matching evaluations.
python -m selection_rule_comparison.collect_score_ranges --plan-only --hours 2
```

Use `--output-dir PATH` to start an independent collection, for example with a
different `--pair-index`, `--sets`, `--wiki-sets` or `--blas-threads` value.

## Analysis notebooks

Open either notebook and run from top to bottom. These notebooks read saved
outcomes and source predictions; they do not import or run the collector, train
models or call matching algorithms.

- [Real-network analysis](../../real_data_experiments/notebooks/02_score_range_banks_real.ipynb)
  covers all eight pair cases, the three frozen references, the synthetic mean
  predictor and every individual source model.
- [Synthetic analysis](../notebooks/06_score_range_banks.ipynb) covers all 27 cells and compares
  one B with the mean over three Bs on identical random sets.

Both include score-range coverage, per-set score/performance scatterplots, named
rule comparisons (including clustered), additional feature correlations,
approximately degree-matched contrasts and solver-convergence diagnostics.
Shared loading, summaries and plotting live in `score_range_analysis.py`.

The notebooks save CSV tables and PNG/PDF figures separately from collection data,
under `selection_rule_comparison/data/score_range_analysis/real/` and
`selection_rule_comparison/data/score_range_analysis/synthetic/`.
The main analyses require complete common evaluation blocks; they do not silently
mix different draw/repeat counts after a partially completed collection.
