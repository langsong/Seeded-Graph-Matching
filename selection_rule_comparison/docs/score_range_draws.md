# Extra draws of B for the synthetic score-range banks (11 draws per set)

Requested on 18 Nov: the fresh-pair view (weighted score against each set's outcome averaged over
draws of B) for the 27 other synthetic settings, across each bank's full score range. The score-range
collection evaluated every synthetic bank on draws 0–2 of B; this add-on evaluates the same 150 seed
sets per bank on draws 3–10, so every set has 11 draws (draw 0 is the saved B).

## Run

From the repository root, in the `ds` environment:

```bash
conda activate ds
python -m unittest selection_rule_comparison.tests.test_score_range_draws               # 6 checks, a few seconds
python -u -m selection_rule_comparison.score_range_draws --smoke --hours 0.2       # real matchers, 12 runs
caffeinate -i python -u -m selection_rule_comparison.score_range_draws --hours 10 --jobs 3
```

From another terminal, at any time (also before starting, for the time estimate):

```bash
python -m selection_rule_comparison.score_range_draws --status
```

- **Size.** 32,400 runs: 27 banks × 150 sets × 8 draws. From the run times recorded for draws 0–2, about
  14 h of solver time on one worker, so about 5 h on three. Allow 6–7 h of wall time.
- **Keep the Mac awake.** `caffeinate -i` does this; keep it on power. `--hours` is a cap: runs in
  progress finish, then the workers stop.
- **Resume.** Rerun the same command. Completed runs are skipped and a row cut off by an interruption is
  removed. Ctrl-C is safe.
- **Order.** Work goes draw by draw (draw 3 for every bank, then draw 4, ...), slowest banks first, so a
  stopped collection leaves banks with equal numbers of draws, give or take one.
- **Workers.** `--jobs 4` is fine on a machine with at least eight performance cores. Keep
  `--blas-threads 2`: it is part of the design, as in the original collection.
- **Smoke run.** One small bank per matcher, draw 3, four sets each, in `tmp/score_range_draws_smoke/`.

## Design

- **Source.** `data/score_range_collection/`, read only: the same 27 synthetic banks and seed sets.
- **Draws of B.** Draws 3–10 come from the same conditional-B generator as draws 1–2
  (`b_replicates.draw_b`, seeded by the pair seed and the draw number). Every job first regenerates draw 1
  and checks it against the hash saved by the original collection, so a changed generator stops the run.
- **Matching.** The original collector's matcher settings and solver-seed rule (one seed per bank,
  algorithm and draw); GOAT keeps each graph family's regularizer. One run per set and draw.

## Outputs

`data/score_range_draws/`:

| File | Contents |
| --- | --- |
| `settings.json`, `environment.json` | Frozen design (source hash, draws, code hashes) and runtime provenance |
| `<bank>/<algorithm>_draw_XX_runs.csv` | One row per set, with the original run files' columns; `run_id` = 1000 × candidate + 10 × draw |
| `<bank>/draw_XX.json` | Hash of the regenerated B and of its true matching |
| `<bank>/*_errors.csv`, `collection_error_draw_XX.txt` | Failures, if any; retried on resume |
| `collection.log` | Progress |
| `analysis/` | Notebook figures (PNG and PDF) and CSV tables |

A changed design, source collection or collection code is refused on resume; use a new `--output-dir`.

## Analysis

Run `notebooks/07_score_range_11_draws.ipynb` from top to bottom (helpers in `score_range_draws_analysis.py`). It reads
saved outcomes only and works on a partly finished collection: within a bank it averages only draws that
every set has finished.

1. Frozen weighted score against each set's mean outcome over draws: one figure per graph model, with the
   matchers as rows and their three cells as columns.
2. Degree alone, in the same layout.
3. Spearman correlation against the number of draws averaged.
4. A summary per cell: correlations, and the gains of the top decile by score, the highest-scoring set and
   the best hand rule over the uniform random controls.

Set `OUTCOME = 'success'` in the first code cell to plot success rates instead of test accuracy.
