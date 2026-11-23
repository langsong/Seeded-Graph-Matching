> Design notes for `strategy_and_active_v2.py`, which was `overnight.py` until 23 Nov 2026 (this file was `OVERNIGHT.md`).
> Its results are still in `results/overnight_v2/`.

# Revised real-network collection

From a terminal, run:

```bash
cd /Users/langsong/Desktop/Seeded-Graph-Matching
conda activate ds
python -u -m real_data_experiments.strategy_and_active_v2 --hours 6.5 --jobs 3
```

Use `--hours 7` for seven hours. The script invokes macOS `caffeinate` while running; keep the computer powered on. It saves progress continuously. At the time limit it stops starting evaluations, lets any in-flight solver finish and save, and exits. Thus the limit can overrun by the duration of a solver call or selection calculation. It can also finish early if all planned work completes.

The default destination is `real_data_experiments/results/overnight_v2/`. Previous `strategy_comparison/` and `active_test/` results are preserved. Do not run the old collectors to collect this revised design.

Check progress from another terminal:

```bash
python -m real_data_experiments.strategy_and_active_v2 --status
```

Repeat the collection command to resume, giving another time allowance. Counts, random seeds, saved weights, and collection-code hashes are checked on resume. Use a new `--output-dir` if changing the design or collection code. The analysis module and notebook edits do not change collection hashes. A lock prevents two collectors from using the same output directory.

Tomorrow, run **notebooks/03_strategy_comparison_v2.ipynb** and **notebooks/04_active_queries_v2.ipynb** from top to bottom. Both now read this new destination and never launch collection. Change `OUTPUT` in the notebooks if you used a custom destination. They tolerate incomplete collection, report coverage, and plot only balanced comparisons. Choose `METRIC = 'accuracy_test'`, `'accuracy_nonseed'`, or `'accuracy_all'` in their setup cells.

## Collection design

- **Independent calibration.** Random-seed pilot outcomes use a separate held-out partition and RNG namespace. Select three budgets per small pair/matcher and four on Wikipedia, close to random accuracies 0.2–0.7. If the tested accuracy range is under 0.1, retain budgets spanning the tested range. Record unsuccessful attempts to find an intermediate regime; low/high accuracy is not automatically an intrinsic floor/ceiling. Wikipedia percolation's pilot extends to 150 seeds.
- **Static confirmation.** All 12 native pairs; five test partitions on small pairs and three on Wikipedia. Hold out 20% of A vertices from every selection policy. Eight randomized selection draws per small-pair partition and three on Wikipedia, with two solver runs for FAQ/GOAT and one percolation run. Top degree is one deterministic set per partition/budget. Duplicate sets share evaluation files, but all policy memberships are retained. Seven core policies: random, top degree, clustered, spread, distinguishing, frozen-ER best-of-1,000, and frozen-ER best-found. Small pairs also include hub-free distinguishing and adjacency-signature entropy. The ER normalization pool has up to 1,000 distinct sets; when fewer sets exist (for example, a one-seed budget on Enron), both reference and screening pools enumerate the entire eligible population; best-found retains four starts, 12 improving-swap steps, and up to 128 proposals per step. No real-outcome model is trained.
- **Active queries.** Enron 130–132, 145–147, 160–162; FAQ and GOAT; initial top-degree or randomized-distinguishing seeds. Ten repeated trajectories per initial rule, distributed across five partitions. Query one eligible vertex at a time from budget 5 to 30. Four policies share the same initial matching: random, distinguishing continuation, ensemble uncertainty, uncertainty times summed uncertainty of A-neighbors. Each round uses 10 feasible matching runs: one barycentric initialization and nine with initialization perturbation 0.2. All policies share solver RNG blocks and the same edge-objective safeguard. Acquisition never uses unrevealed truth. The safeguard may preserve the corrected previous assignment; uncertainty always comes from fresh runs under current answers.
- **One-query diagnostics.** At initial budgets 5, 10, 20 on the three Enron pairs, evaluate every eligible additional seed, with five repeated initial states per pair/matcher/budget. For Wikipedia, use FAQ, budgets 10 and 20, two initial states each, and 16 mixed candidate vertices. Starting sets use randomized distinguishing. Compare each candidate against a fresh no-query baseline using three paired barycentric solver runs; the initial uncertainty ensemble is separate (eight runs on Wikipedia). These are ground-truth diagnostics, not deployable oracle selectors. Candidate lists are frozen before outcomes are evaluated.

Small diagnostics, small confirmation, active trajectories, and Wikipedia diagnostics have time reservations (up to 30, 45, 45, and 45 minutes respectively). Unused time immediately carries forward. Wikipedia confirmation receives the remaining allowance; if it finishes, remaining small tasks and diagnostics are resumed. No deliberate waiting is used to fill the allowance. At most three small-pair workers or two Wikipedia workers run concurrently, with two BLAS threads each.

## Outputs and interpretation

- `settings.json`, `environment.json`: fixed design, source hashes, environment.
- `calibration/`: independent random outcomes and selected budgets.
- `strategy/<case>/splitXX/<algorithm>/kXXX/`: selection plan and per-evaluation NPZ files.
- `active/<case>/<algorithm>/<initial_rule>/repXX/`: common initial state, fresh ensembles, and policy trajectories.
- `diagnostic/<case>/<algorithm>/kXXX/repXX/`: frozen candidate list, no-query baseline, individual branches, and paired gains.
- `completed/`, `run_status.json`, `errors/`: task completion, run status, and error details. Failed tasks can be retried; successful retries clear their error record.
- `analysis/`: notebook-generated CSVs and PNG/PDF plots.

NPZ records contain assignments, per-vertex correctness, seeds, and fixed test indices. Comparisons save fixed-test, all-nonseed, and total accuracy. Means first average draws within partitions, then weight partitions equally. Plot bands describe variability, not independent-network confidence intervals. Native graph pairs are fixed: partitions and relabelings are not new graph realizations. Full worm and worm LCC overlap, as do Enron pairs sharing weeks.

Only selection draws with all methods and solver repetitions complete enter the strategy plots. Active plots require all four policies at a repetition/budget. Diagnostic summaries require all candidates in a state's frozen list to finish. Raw partial work is retained and resumes without rerunning completed evaluations.

## Small validation run

```bash
python -m unittest real_data_experiments.tests.test_strategy_and_active_v2 real_data_experiments.tests.test_active_test real_data_experiments.tests.test_strategy_comparison
python -u -m real_data_experiments.strategy_and_active_v2 --smoke --hours 0.1 --jobs 2
```

The smoke design uses its own `results/overnight_v2_smoke/` directory and exercises actual FAQ, GOAT, and percolation calls, calibration, strategy plans, active trajectories, and diagnostics on one small Enron pair.
