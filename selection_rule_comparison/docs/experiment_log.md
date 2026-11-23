> Dated log of the experiments in this folder, moved here from `README.md` on 23 Nov 2026. Paths and commands are as
> they were when each section was written; `../README.md` says where things are now (retired experiments are
> in `archive/`, notebooks in `notebooks/`).

# Seed-selection rule comparison (round 1)

Do strategies built from the feature diagnostics need fewer queries than top degree? This folder runs the round-1 comparison from the project note `next-steps-plan-2026-10-20`. The predictions it tests are written down in `core-panel-refit-predictions-2026-10-20` and in `fixed_pair_seed_set_classification/core_panel_refit.ipynb`.

**GOAT temperature audit:** open `temperature_audit.ipynb` with the `ds` kernel and run all cells. It fixes round-1 graph pairs and top-degree/hub-free seed sets while comparing regularizers 100, 300 and 600. The default is 120 calls (five pairs per model, Chung–Lu and PAPER, two budgets, two rules), with resumable checkpoints under `data/temperature_audit/`. It records matching accuracy, gradient scale, transport entropy, inner/outer convergence and numerical warnings. A separate output folder can be used for the short trial described in the notebook. `temperature_audit.py` provides its instrumented runner without changing the production solver.

Round 1 is a proof of concept with 20 fresh graph pairs per graph model. More pairs can be added later without redoing anything (see **Adding pairs**).

## Design
- **Graph pairs.** Every strategy runs on the same fresh, structure-checked graph pairs. Pairs are drawn from the settings of the training collections but are never the training or calibration pairs.
- **Strategies are orderings.** A strategy ranks Graph A's vertices once, and the seeds at budget k are its first k vertices. Sets are therefore nested across budgets, like buying queries one at a time.
- **Common random numbers.** Solver seeds depend only on (pair, budget, restart), so every strategy sees the same solver randomness. If two strategies pick the same seed set at a budget, the stored result is reused.
- **Success labels.**
  - FAQ and GOAT: unseeded accuracy of at least 0.90.
  - Percolation: accuracy of at least 0.50, the upper mode recommended on 7 Oct. Accuracy of at least 0.90 is also recorded.
  - FAQ restarts: up to 3 per budget, stopping at the first success. Because the objective flags success, this equals best-of-3.

| Strategy | Picks | Run for |
|---|---|---|
| `random` | random ordering | all |
| `top_degree` | highest degree (the bar) | all |
| `top_harmonic` | highest harmonic centrality | percolation |
| `clustered` | degree-first, maximise pair mass | all |
| `spread` | degree-first, maximise coverage | all |
| `distinguishing` | degree-first, maximise signature uniqueness (cap 3, then cap 2) | all |
| `hub_free_distinguishing` | `distinguishing` with Graph A's top 10 hubs removed from the pool | GOAT |

"Degree-first" means seed j is chosen from the 2j highest-degree unused vertices, so every set stays close to top degree.

| Algorithm | Graphs (as in the training collections) |
|---|---|
| `percolation` | 7 Oct settings, mean degree about 12 (ρ 0.7 / 0.62 / 0.7), ExpandWhenStuck, r = 2 |
| `goat` | 13 Oct settings, mean degree 16 (ρ 0.5 / 0.4 / 0.45), regulariser 100 / 300 / 300 |
| `faq` | the same graphs as GOAT (identical pairs) |

**Budgets.** Each graph model has its own grid of six budgets, set from the 20 Oct pilot (random and top degree on 10 pairs). A grid starts where both baselines still mostly fail, so a strategy that beats top degree has room to show it, and ends where random mostly succeeds.

| Algorithm | ER | PAPER | Chung–Lu |
|---|---|---|---|
| Percolation | 3 4 5 6 8 10 | 2 3 4 6 8 10 | 2 3 4 5 6 8 |
| GOAT | 6 8 9 10 12 16 | 6 8 9 10 12 16 | 4 6 8 10 14 20 |
| FAQ | 16 18 20 22 24 28 | 14 16 18 20 24 32 | 12 15 18 21 24 28 |

All settings live in `settings.py`. At start-up the runner checks the graph settings against the training presets; the result is stored in `metadata.json`.

## Running
Run from the repository root in the `ds` environment. One command runs everything:

```bash
python -m selection_rule_comparison.run_comparison
```

- The algorithms run one after another (percolation, GOAT, FAQ). Within an algorithm, the three graph models run in parallel, each limited to 3 BLAS threads.
- The terminal shows when each model starts and finishes. Per-pair progress, with an estimate of the time left, goes to `selection_rule_comparison/logs/main_<algorithm>_<model>.log`. To follow one, run `tail -f` on it in a second terminal.
- On a Mac the run keeps the computer awake until it finishes (`caffeinate`; turn this off with `--no-caffeinate`). Closing a laptop's lid still puts it to sleep.
- If the run stops (an error, Ctrl-C or a closed terminal), run the same command again. Finished pairs are kept, and the run continues from the next pair.

**Approximate time.** These estimates come from the pilot's run times on the same Mac and could be off by about 30% either way.

| Algorithm | 20 pairs (default) | `--pairs 10` |
|---|---|---|
| Percolation | about 5 min | about 3 min |
| GOAT | about 35 min | about 18 min |
| FAQ | about 35 min | about 18 min |
| All three | about 1¼ h | about 40 min |

**Options**

| Option | Effect |
|---|---|
| `--pairs 10` | 10 pairs per model instead of 20 |
| `--algorithm goat` | one algorithm: `percolation`, `goat` or `faq` |
| `--model chung_lu` | one graph model |
| `--jobs 1` | run the models one after another, with progress in this terminal |
| `--budgets …`, `--strategies …` | override the grid or the strategies; use a new `--output-root` |
| `--overwrite` | delete the selected results and start them again |

**Adding pairs.** Pair i is the same graph pair whatever `--pairs` is. Running `--pairs 10` and later the plain command (20 pairs) therefore gives exactly the results of running 20 at once, and the second run computes only pairs 11–20. A larger `--pairs` later works the same way. The budgets and strategies must stay the same, because the runner refuses to mix settings in one folder.

**Pilot (done 20 Oct).** `--phase pilot` runs random and top degree on 10 pairs per model over a wide grid. Section 1 of the notebook plots it. The budget table above was set by hand from those curves and replaces the commands that section prints.

## Results
Set `PHASE = 'main'` in `rule_comparison.ipynb` and run all cells. The notebook uses the pairs finished so far, so it can be run while later algorithms are still going. Set `SAVE_FIGURES = True` to write the figures to `figures/round1/`.

With 20 pairs the bands are wide, and only large differences will be clear. For example, with 20 untied pairs the sign test on k\* needs 15 wins to reach p < 0.05 (9 of 10 with 10 pairs). Treat round 1 as a check of direction.

## Outputs
Results are written to `data/round1/<phase>/<algorithm>/<model>/`:

| File | Contents |
|---|---|
| `settings.json` | Graph, solver, budget, strategy and seed-stream settings. A rerun refuses to mix different settings into the same folder. |
| `metadata.json` | Status, environment (package versions, threads, git commit), training-preset check and timing |
| `pairs.csv` | One row per completed pair: pair seed, rejected draws, degrees, kept-edge fraction, hashes of A and B, seconds |
| `orderings.csv` | Each strategy's ordering, up to the largest budget, for every pair |
| `runs.csv` | One row per solver run: strategy, budget, restart, solver seed, seed vertices, accuracy, success labels, objective, iterations, runtime, whether it was reused (`cached`), and the core features of the seed set (`f_*`) |

Progress logs are in `logs/<phase>_<algorithm>_<model>.log`; each run appends to them.

A pair is written only once it is complete, so an interrupted run resumes at the next pair. Use `--overwrite` to restart one model, or a new `--output-root` to keep old results.

**Reproducibility.**
- Pair seeds come from a fixed stream per graph family, model, phase and pair index. The pilot and main runs therefore use different pairs, and FAQ and GOAT use identical pairs.
- Strategy tie-breaks and the random ordering use a stream per pair and strategy.
- Graph pairs are not stored, but each one is regenerated exactly from `pair_seed` with `experiment_common.graphs.generate_graph_pair`. The stored hashes confirm the match.

## Follow-up experiments (27 Oct)

Each has a script and a notebook of the same name. Run a notebook from the top: its run cell starts the experiment (or skips work already done), and the rest analyses it. The scripts can also be run from the repository root, for example `python -m selection_rule_comparison.hub_test`.

| Notebook | Question | Time |
|---|---|---|
| `hub_test.ipynb` | GOAT: hubs or uniqueness, wrong hints or too broad, are hubs harmful or wasted; and does skipping hubs help FAQ or percolation? Reuses round-1 pairs. | about 25–35 min (the first two GOAT tests are already done) |
| `regime_sweep.ipynb` | When does skipping hubs help GOAT? Fresh Chung–Lu pairs over tail weight (α = 2, 3, 6) and noise (ρ = 0.40–0.55). | about 45–75 min |
| `borderline_check.ipynb` | Does clustered beat top degree for percolation (50 fresh pairs on ER and PAPER)? Is degree all that matters for FAQ (round 1 extended to 20 pairs)? | about 45 min |
| `real_graph_check.ipynb` | Does skipping hubs help GOAT on the Enron email network, and does it help FAQ or percolation? | about 10 min |

Suggested order: `hub_test`, then `regime_sweep`, `borderline_check` and `real_graph_check`. After `borderline_check` adds FAQ pairs, rerunning `hub_test` adds its FAQ test on the new pairs. Results go to `data/hub_test/`, `data/regime_sweep/`, `data/borderline/`, `data/real_graph/` and, for FAQ, round 1's own folder.

## Beyond top degree (28 Oct)

`beyond_top_degree.ipynb` (script `beyond_top_degree.py`) asks whether the best rule sits near top degree or somewhere else, and whether a feature helps on its own or only through degree. It reuses existing pairs, so round 1, `hub_test` and `borderline_check` results are compared pair by pair.

- **Path away from top degree.** Seed j is chosen among the m·j highest-degree unused nodes, for m = 2 (round 1), 8 and all nodes (percolation adds 4 and 16). This is done for pair mass, uniqueness (fingerprint entropy; for FAQ, unique fingerprints first) and the frozen model's score. Percolation also gets a witness score with no fitted weights.
- **Look-alikes.** Each all-node rule gets random nodes with the same degrees. Rule − look-alike is what the feature adds; look-alike − random is what the degree level buys.
- **Checks.** Top degree is rerun on pairs 0, 1, 100 and 101 and must match the stored runs. The frozen model at 2j must match the 21 Oct pilot.

| Arm | Pairs | Time |
|---|---|---|
| Percolation | round-1 pairs 0–19 (all models) and `borderline_check`'s fresh pairs 100–149 (ER, PAPER) | about 45 min |
| GOAT | round-1 pairs at round 1's regulariser, and again at the other one (ER 300, PAPER 100, Chung–Lu 100) | about 2 h |
| SGM (FAQ) | round-1 pairs, ER and PAPER | about 2 h |

Run the notebook from the top (set `ARMS_TO_RUN` to run fewer arms), or from the repository root: `python -m selection_rule_comparison.beyond_top_degree --arms percolation goat goat_other faq`. Results go to `data/beyond_top_degree/<arm>/<model>/pair_XXX.csv`, with the frozen model's weights in `data/beyond_top_degree/weights/` and a settings snapshot in `data/beyond_top_degree/settings.json`. Reruns resume.

## Best of N (2 Nov)

`best_of_n.ipynb` (script `best_of_n.py`) tests a screening rule on round 1's pairs. The rule: draw N random seed sets of the budget's size, score each one from graph A with the pooled weights from `fixed_pair_seed_set_classification/multi_pair_models.ipynb`, and query the highest-scoring set. None of round 1's pairs was used to fit the weights, so round 1's stored results for random, top degree and the other rules are compared pair by pair.

- **Score.** A weighted sum of degree, harmonic, pair mass and uniqueness (cap 3; cap 2 for FAQ). Each feature is measured in SDs of the random sets drawn on the same graph at the same budget. The weights, one set per algorithm and graph model, come from `fixed_pair_seed_set_classification/data/multi_pair_2026-10-30/pooled_weights.json`.
- **New rules.** Best of 100 and best of 1,000 by the score (the 100 are the first of the 1,000 draws), and best of 1,000 by degree alone, as a control.
- **Runs.** Round 1's solver settings and solver seeds. FAQ restarts up to 3 times and stops at the first success, as in round 1.
- **Checks.** Every pair is checked against round 1's hashes. Round 1's top-degree sets are rerun on pairs 0 and 1 and must match.

It takes about 1.5 hours with three pairs at a time: percolation about 5 min, GOAT about 25 min, FAQ about 1 h. Run the notebook from the top, or from the repository root: `python -m selection_rule_comparison.best_of_n` (`--status` shows progress). Results go to `data/best_of_n/<algorithm>/<model>/`: `pair_XXX.csv` holds the solver runs, and `pair_XXX_scores.csv` the score of every screened and round-1 set. A settings snapshot goes to `data/best_of_n/settings.json`. The notebook's last section draws a draft NetSci figure into `data/best_of_n/figures/`. Reruns resume.


### Best of N: additions (2 Nov, later)

- **Section 7** of `best_of_n.ipynb` repeats section 10 of `rule_comparison.ipynb` with best of 100, best of 1,000 and best of 1,000 by degree added. It shows one small marker per rule's seed set per pair, with degree and with the pooled score against final accuracy, and each rule's mean over pairs as a large marker with 95% intervals. Each setting uses the budget where mean success is closest to 50%, which can be changed. A paired view compares each rule with top degree on the same pair. It reads saved results only.
- **GOAT at r = 100** (section 8) is a robustness check. Run it with `run_r100()`, the notebook's first run cell, or `python -m selection_rule_comparison.best_of_n --r100`. It runs GOAT on PAPER and Chung–Lu at graspologic's default regulariser, for round 1's rules and the screened rules. Pairs, sets, budgets and solver seeds are round 1's, and the screened sets are checked against those saved at r = 300. It takes about 40 minutes with 3 jobs. Results go to `data/best_of_n/goat_r100/<model>/`. r = 300 stays the main setting.


## Score transfer (2 Nov)

`score_transfer.ipynb` (script `score_transfer.py`) asks how well the pooled score from `fixed_pair_seed_set_classification/multi_pair_models.ipynb` ranks seed sets outside the data it was fitted on. The weights are never refitted; each setting keeps its own.

- **Part A (plan 1): new pairs.** Round 1's 20 pairs per setting, at the training budget. Each pair gets 60 random sets and 40 top-scoring sets, plus round 1's rules at the same budget. The success-by-percentile curve is compared with the multi-pair held-out curve, and round 1's rules are placed on the same score. The high-scoring sets also trace best of N from N = 20 to 20,000: the set ranked r-th of 20,000 stands for best of 20,000 / r.
- **Part B (plan 2): other budgets.** Round 1's pairs 0–9, at the training budget minus and plus 2 seeds (percolation) or 3 seeds (GOAT, FAQ). Each pair and budget gets 36 random and 24 top-scoring sets.
- **Part C (plan 2, redesigned 2 Nov): other sizes and sparsities of the same graph model.** A grid of graph size (300, 600 and 1,200 vertices) by mean degree (× 0.75, × 1 and × 1.5 of round 1's: 12 for percolation, 16 for GOAT and FAQ), on 10 fresh pairs per cell. Round 1's setting (600 vertices, × 1) is part A. Percolation runs every cell. GOAT and FAQ run 300 and 600 vertices at all three densities, and 1,200 vertices only for Chung–Lu at × 1, because their runs are several times slower there. The model's shape, ρ and the solver settings stay as in round 1. Mean degree is set through the ER edge probability, PAPER's background probability or the Chung–Lu weight scale.

  Each cell first calibrates its budget, so that random sets succeed about half the time (30 random sets per budget tried). A cell where no random set succeeds even at the top of the budget ladder is reported and skipped. Each pair then gets 36 random and 24 top-scoring sets, scored with the base setting's weights.
- **Sets.** On each pair and budget, 20,000 random sets are drawn and scored. The random sets are the first ones drawn. The top-scoring sets are spread over three bands of the rest: the top 0.1% (including the best set), 99–99.9% and 95–99%.
- **Runs.** One run per set, with round 1's solver settings. Each SGM run gets its own solver seed. Round 1's pairs are checked against its hashes, and round 1's rule sets against its stored orderings.

It takes about 12 hours with 3 jobs: part A about 3.5 h, B about 2 h and C about 6.5 h. Run the notebook from the top, or from the repository root: `python -m selection_rule_comparison.score_transfer`. To split it over two nights, use `--parts A B` and then `--parts C`; `--status` shows progress. Reruns resume. The redesign of part C left parts A and B unchanged, so part-A pairs finished before it are kept (their seed sets are reproduced exactly).

Results go to `data/score_transfer/`:
- `A/<algorithm>/<model>/pair_XXX.csv`, `B/<algorithm>/<model>/<k0-d or k0+d>/pair_XXX.csv` and `C/<algorithm>/<model>/<cell>/pair_XXX.csv` hold one row per seed set: its group, band, score, percentiles, features and outcome.
- A part-C cell is named like `n300_x0.75` (300 vertices, mean degree × 0.75). Part C also writes `calibration.json` and `calibration/kXX.csv` for every cell.
- `settings.json` holds the design. A rerun with a different design refuses to mix results in the same folder; set `SCORE_TRANSFER_OUTPUT_DIR` to use another one.


## Rules on graphs with communities (SBM, 2 Nov)

`sbm_rules.ipynb` (script `sbm_rules.py`, graphs in `sbm_graphs.py`) asks whether round 1's general rules still beat random on stochastic block models. The pooled weights never saw communities, so only the rules are tested here. SBM's own weights come from `fixed_pair_seed_set_classification/sbm_models.ipynb`.

- **Graphs.** Two kinds, each matched to a round-1 model so that communities are the only change: plain SBM ("ER plus communities") and degree-corrected SBM ("Chung–Lu plus communities"). Both have 4 equal blocks of 150 vertices, with 80% of edges inside blocks. Mean degree, ρ and GOAT's regulariser are the matched model's: percolation has mean degree 12 and ρ = 0.70; GOAT and FAQ have mean degree 16 and ρ = 0.50 (SBM) or 0.45 (degree-corrected); GOAT's regulariser is 100 (SBM) or 300 (degree-corrected). Variants with unequal blocks (300/150/100/50) or weaker communities (60% of edges inside) are defined in `sbm_graphs.SETTINGS` (`sbm_unequal`, `dcsbm_unequal`, `sbm_weak`, `dcsbm_weak`) and can be run later by name.
- **Budgets.** Each algorithm and setting first calibrates k50, the budget where a random set succeeds about half the time. It runs 3 random sets on each of 10 separate calibration pairs per budget tried, starting from the matched model's training budget. The grid is k50 × (0.4, 0.55, 0.7, 0.85, 1.0, 1.25) for percolation, × (0.4, 0.55, 0.65, 0.75, 0.9, 1.1) for GOAT and × (0.55, 0.65, 0.75, 0.85, 0.95, 1.1) for FAQ, rounded. It is extended upward to at least 5 budgets when k50 is small.
- **Rules.**
  - Round 1's rules for each algorithm.
  - Top degree by block: top degree spread over the blocks in proportion to their sizes, from the true labels. This is a first test of whether spreading seeds over communities helps; a practical version would infer the blocks from graph A.
  - Best of 1,000 by the pooled score, with the ER, PAPER and Chung–Lu weights averaged (for reference only).
  - Best of 1,000 by degree.
- **Runs.** 20 fresh pairs per setting, with round 1's protocol: FAQ restarts up to 3 times, and solver seeds depend only on the pair, budget and restart. Each row records the blocks the seeds cover and the largest block's share of the seeds.
- **Direct test of SBM's own weights (added 3 Nov).** Once `fixed_pair_seed_set_classification/sbm_models.ipynb` has fitted and saved SBM's own weights, `run_sbm_weights()` runs best of 100 and best of 1,000 with them on these 20 test pairs per setting, at the same budgets. They choose from the same 1,000 random draws as the averaged-weight best of 1,000 and meet the same solver seeds, so the comparison isolates the weights. A set already run at the same budget is not run again. It takes about 25 minutes. `sbm_models.ipynb` runs it in section 7, or from the repository root: `python -m selection_rule_comparison.sbm_rules --sbm-weights`.

It takes about 1.5 hours with 3 jobs. Run the notebook from the top, or from the repository root: `python -m selection_rule_comparison.sbm_rules`. Use `--settings sbm` to run one setting, and `--status` to see progress. Results go to `data/sbm_rules/<algorithm>/<setting>/`:
- `calibration.json` and `calibration/kXX.csv` hold the budget calibration.
- `pair_XXX.csv` holds the runs.
- `settings.json` holds the settings.
- `sbm_weights/pair_XXX.csv` holds the direct test's runs, and `sbm_weights/pair_XXX_scores.csv` every rule's set on SBM's own score.

Reruns resume. The calibrations are shared with `fixed_pair_seed_set_classification/sbm_collection.py`.

## Dense fresh-pair score plots (3 Nov)

`fresh_pair_scores.ipynb` and `fresh_pair_scores.py` plot the frozen pooled score against actual unseeded matching accuracy. The default is 3 fresh pairs per algorithm/graph setting, 1,000 distinct uniform random seed sets per pair, and one matching run per set: 27,000 runs, roughly 5–7 hours with 3 workers while other matching collections are stopped. Graph parameters and budgets match the pooled training collection; no recalibration or model fitting takes place. FAQ runs once per set.

Run from the repository root in the ds environment:

```bash
python -m selection_rule_comparison.fresh_pair_scores
python -m selection_rule_comparison.fresh_pair_scores --status
python -m selection_rule_comparison.fresh_pair_scores --plot-only
```

Every completed set is saved and flushed immediately, so Ctrl-C stops the workers and the same command resumes. Results go to `data/fresh_pair_scores/<algorithm>/<model>/pair_XXX/sets.csv`; graph hashes, feature normalization and frozen weights are recorded alongside them. A settings snapshot prevents mixing designs. `FRESH_PAIR_SCORES_OUTPUT_DIR` selects a separate output folder for a different set count or a trial. GOAT and FAQ share fresh graph pairs. The new pair stream excludes known fixed, training, round-1 and completed grid-test pairs.

Figures are saved as PNG and PDF in `data/fresh_pair_scores/figures/`, with a separate panel per pair, grey random-set points and distinct markers for any added selected sets. `summary.csv` records within-pair Spearman correlation and top/bottom fifth mean accuracy. The logistic models were trained on threshold success; these plots deliberately show continuous accuracy instead.

**Plot update (10 Nov).** The scatter plots now use points only; the orange score-decile curves and the B-redraw plots' orange bin averages have been removed. The frozen score is the weighted sum of four feature z-scores, centered/scaled against each pair's original 1,000 random sets. Its mean over those sets is zero, but its SD is not necessarily one. It is not a probability, and the two accuracy bands need not form a linear trend.

`python -m selection_rule_comparison.fresh_pair_scores --selected` adds, on the same saved graph pairs and at the same budgets:

- the ten sets with the largest seed-to-nonseed degree in a new pool of 20,000 uniform sets;
- the ten highest frozen-score sets in that pool;
- the existing round-1 constructed rules, including the top-degree-vertex rule.

Selections use A only. Every added set uses the original feature means, SDs and weights; no normalization or model is refitted. The pool tails are finite-pool optima, while the constructed rules build sets directly. Identical sets reuse a saved outcome. Results resume in `selected_sets.csv` and `selection.json` beside each pair's original data, without changing `sets.csv`. There are 666 selected-set rows across the 27 panels, taking roughly ten minutes with three jobs. The notebook's `RUN_SELECTED` switch runs this extension before plotting. Spearman and `summary.csv` remain random-only. The B-redraw section still uses its original 100 sets per pair; the selected-set extension runs on the original B only.

**Metric extremes and wider score coverage (10 Nov).** `fresh_pair_extremes.py` adds exact global top-ten sets for mean total vertex degree and mean harmonic centrality. A best-first heap enumerates the best fixed-size combinations of sorted vertex weights. It also adds ten best-found sets for each of cut degree, pair mass, uniqueness, coverage and the frozen score, using three search starts, up to eight improving single swaps and 256 sampled swap candidates per step. Any vertex can enter; these search results have no general global-optimality guarantee. Original random sets and the earlier selected sets are included among the candidates. Total vertex degree differs from the score's cut-degree feature by twice the internal seed-edge count divided by k.

To fill the gap between random and near-optimal sets, it builds swap paths from a random set towards each metric's best set and takes up to two intermediate sets from each of ten frozen-score intervals. Results go to `metric_sets.csv` and `metric_selection.json`; earlier data remain available. For plotting, the non-additive metric groups are reranked across the pooled evaluated candidates from every search, with objective ties broken by frozen score. This allows one search to contribute a better set to another metric's plotted group; the CSV retains the original generating groups. Plots prefer this wider collection where available and label exact global ranks separately from best-found heuristic sets. The notebook's `RUN_METRICS` switch collects missing results. From the repository root:

```bash
python -m selection_rule_comparison.fresh_pair_extremes --jobs 3
```

`--compare-models` reads existing score-transfer A results, refits the old four-feature models on the original fixed pairs' 600 random sets, and compares them with the frozen pooled weights on the same 20 external pairs and 60 random sets per pair. It saves weight, performance and per-pair AUC tables in `data/fresh_pair_scores/model_comparison/`, without running more matching evaluations. Old coefficients are expressed in the same raw standardized feature coordinates as the pooled coefficients. The old and new training budgets can differ; this compares the existing models rather than isolating only the number of training pairs.


## Graded outcomes (4 Nov)

`graded_outcomes.ipynb` (script `graded_outcomes.py`) looks for settings where single runs end part-way instead of all or nothing, then asks whether the frozen pooled score tracks the match ratio there. At round 1's settings only 0–6% of runs end between 0.35 and 0.85 accuracy. It uses round 1's ER pairs for GOAT and FAQ (600 vertices, mean degree 16) with a lower edge correlation ρ; round 1 uses 0.50. A rough check with scipy's FAQ found graded runs at ρ = 0.3: six runs with 150 random seeds ended between 0.44 and 0.68 accuracy.

- **Pilot.** ρ = 0.25, 0.30 and 0.35. FAQ budgets are 60, 90, 120, 150, 180, 220 and 260; GOAT budgets are 20, 35, 50, 70, 100, 140 and 200. Each budget gets 24 random seed sets on one pair per ρ, one run each. `recommend()` picks, for each algorithm, the setting where at least 75% of runs end with test accuracy between 0.2 and 0.8 and the mean is closest to 0.5. An algorithm with no such setting is skipped.
- **Collection.** In the chosen setting, 3 fresh pairs × 1,000 random sets, one run each, with all 11 graph-A features. The first 50 sets of each pair are run twice more with other solver seeds, to measure how much of the spread is solver luck.
- **Outcome.** With 100 or more seeds, the choice of seeds also changes which vertices non-seed accuracy counts. So each pair sets aside 120 fixed test vertices that are never seeds, and the main outcome is the share of them matched correctly. Every seed set is judged on the same vertices. Non-seed accuracy is recorded too.
- **Runs.** graspologic FAQ or GOAT with round 1's solver settings (GOAT regulariser 100). Each run has its own solver seed. Pairs come from a new seed stream that skips every pair seed used so far; FAQ and GOAT share pairs at the same ρ.
- **Analysis.** The frozen score, degree alone and single features against the match ratio, within each pair. Then weights refitted on the match ratio (ridge regression, one pair left out at a time), with the 4 core features and with all 11.

The pilot takes about 30 minutes and the collection about 2 hours with 3 jobs; both are rough guesses. Run the notebook from the top; set `RUN_COLLECTION = False` to stop after the pilot, or set `CHOICE` to pick the setting yourself. From the repository root:

```bash
python -m selection_rule_comparison.graded_outcomes --pilot
python -m selection_rule_comparison.graded_outcomes --collect
python -m selection_rule_comparison.graded_outcomes --collect --choice faq=0.30:150 goat=0.30:70
python -m selection_rule_comparison.graded_outcomes --status
```

`--collect` without `--choice` uses the pilot's recommendation. Reruns resume. Results go to `data/graded_outcomes/`:
- `pilot/<algorithm>/rho0.30_k150.csv` holds one row per run; the `.json` next to it records the pair, its test vertices and the feature means and SDs.
- `collection/<algorithm>/rho0.30_k150/pair_XXX/` holds `sets.csv` (one row per set, with every feature), `repeats.csv` and `pair.json`.
- `settings.json` holds the design. A rerun with different weights or set counts refuses to mix results in the same folder; set `GRADED_OUTCOMES_OUTPUT_DIR` to use another one.

### Graded outcomes, section 8: rules at low ρ (5 Nov)

Section 8 of `graded_outcomes.ipynb` (`graded_outcomes.py --rules`) asks whether round 1's rules and best of N still beat random seed sets when runs end part-way. It uses 5 new pairs per ρ, shared by FAQ and GOAT, with 5 budgets each:

| ρ | FAQ budgets | GOAT budgets |
|---|---|---|
| 0.25 | 120, 150, 180, 220, 260 | 100, 140, 170, 200, 240 |
| 0.30 | 90, 120, 150, 180, 220 | 70, 100, 120, 140, 170 |
| 0.35 | 40, 50, 60, 70, 80 | 30, 40, 50, 60, 70 |

Per pair and budget it runs:
- **Random sets:** 10, one run each.
- **Round 1's rules:** top degree, clustered, spread and distinguishing, plus hub-free distinguishing for GOAT. Built as in `strategies.py`, but never picking a test vertex.
- **Best of N:** from one draw of 1,000 random sets, the best of the first 100 and the best of all 1,000 by the frozen score, and the best of all 1,000 by degree.

FAQ runs each rule or best-of set 3 times and averages the test accuracy. GOAT runs it once, because it gives the same result for the same seeds. The outcome is accuracy on each pair's 120 test vertices. In all, 3,675 runs, about 40 minutes with 3 jobs.

Results go to `data/graded_outcomes/rules/<algorithm>/rho0.30/pair_XXX/`. `runs.csv` has one row per run, with the set's core features, its z-scores against the 1,000 random sets, and the frozen score; `pair.json` sits beside it. `rules/settings.json` freezes the design.

```bash
python -m selection_rule_comparison.graded_outcomes --rules
```

## B replicates for the fresh pairs (5 Nov)

`b_replicates.py` (the last section of `fresh_pair_scores.ipynb`) draws B again given A for the fresh pairs, so each seed set gets a success rate instead of a single 0 or 1. On each pair it takes 100 of the 1,000 sets, 10 from each tenth of the frozen score. Each set runs once on each of 10 new draws of B, and the saved B counts as an 11th draw.

**How new Bs are drawn.** They come from the pair's own model. A vertex pair has an edge with probability P + ρ(1 − P) where A has an edge, and P(1 − ρ) where it has none. P is:
- p, for ER;
- the pair's own Chung–Lu probabilities, recorded while the pair is regenerated;
- p away from the shared tree, for PAPER, whose B keeps the tree.

Each new B must pass the same structure check (connected, no automorphisms), and its labels are shuffled. All sets on a pair meet the same draws, and GOAT and FAQ share them.

**Check.** The check covered all 18 distinct saved pairs (GOAT and FAQ share 9), comparing each saved B with 20 new draws. The saved B's edge count, kept-edge fraction and likelihood under this model fell within the new draws' range in 53 of 54 cases.

**Running it.** 27,000 runs, about 5–6 hours with 3 jobs; pair 0 of every setting runs first.

```bash
python -m selection_rule_comparison.b_replicates
python -m selection_rule_comparison.b_replicates --algorithms faq
python -m selection_rule_comparison.b_replicates --status
```

**Results** go to `data/fresh_pair_scores/b_replicates/<algorithm>/<model>/pair_XXX/`:
- `runs.csv`: one row per run, with the draw number.
- `draws.json`: the chosen sets, and each draw's hash and number of rejected draws.

`summary_table()` splits single-run outcomes into a seed-set part, a draw-of-B part and the rest. It also compares Spearman with single runs against Spearman with success rates.

## Regime map (6 Nov)

`regime_map.ipynb` (script `regime_map.py`) maps where single runs end all-or-nothing, where the match ratio is spread out, and where seed choice matters, before any large data collection. Everything before this used calibrated settings where random sets succeed about half the time.

**Grid:**
- 600 vertices, mean degree 8, 16 and 32, and ρ = 0.2, 0.3, 0.4, 0.5, 0.7 and 0.9.
- One model at a time: ER first, then Chung–Lu, PAPER later. Chung–Lu uses mean degree 10 instead of 8: at 8, only 12–25% of drawn pairs are connected with no automorphisms, against 70% for ER.
- Percolation, FAQ and GOAT run on the same pairs.

**Outcome:** accuracy on 120 fixed test vertices per pair that are never seeds, with non-seed accuracy and round 1's success flag recorded alongside.

**Stage 1 (random sets only):**
- Budgets 3 to 300 seeds, each about 1.5× the last, on 2 pairs per cell.
- 8 sets per budget, cut to 4 when they all end below 0.05 or all above 0.95. Climbing stops after two budgets in a row average above 0.95.
- At the budget closest to 0.5:
  - FAQ and GOAT rerun 5 sets twice more, to measure solver luck;
  - GOAT runs 8 sets with its regulariser halved and doubled;
  - the features of 1,000 random sets give their spread.
- About 1.5 hours for ER with 3 jobs (a rough guess).

**Stage 2:**
- Covers every cell whose curve rises across 20–80% inside the range, at the budgets where random sets average 20%, 50% and 80%, on 4 pairs (the stage 1 pairs plus 2 new ones).
- Per pair and budget: 10 random sets; round 1's rules, never seeding a test vertex; best of 100 and 1,000 by the frozen score; best of 1,000 by degree.
- FAQ runs each rule or best-of set 3 times; GOAT and percolation run once.
- Up to about 3 hours for ER.

From the repository root:

```bash
python -m selection_rule_comparison.regime_map --stage 1 --model sparse_er
python -m selection_rule_comparison.regime_map --stage 2 --model sparse_er
python -m selection_rule_comparison.regime_map --status --model sparse_er
```

Results go to `data/regime_map/<model>/` and reruns resume:
- `settings.json` freezes the design.
- `d16_rho0.30/<algorithm>/stage1/pair_XXX/` holds `runs.csv` (one row per run) and `pair.json` (the pair, its test vertices, the budget closest to 0.5 and the feature spread).
- `stage2/budgets.json` fixes the stage 2 budgets once stage 1 is done.

## Match-ratio landscape (7 Nov)

`landscape.py` collects single-run match ratios together with seed-set features in cells where outcomes are graded, for ER, Chung–Lu and PAPER. It answers the meeting's question of how match ratio tracks the features when runs are not all-or-nothing, and it includes cells where seed choice barely pays.

**Cells.** From the ER regime map, one cell per algorithm where round 1's rules gained a lot over random sets, one moderate, and one small. Chung–Lu uses mean degree 10 instead of 8, as in its map.

| Algorithm | Cells (mean degree, ρ) | ER budgets (50% level) |
|---|---|---|
| Percolation | (8, 0.7), (16, 0.4), (32, 0.3) | 13, 51, 93 |
| FAQ | (32, 0.3), (8, 0.5), (16, 0.3) | 71, 58, 143 |
| GOAT | (8, 0.5), (8, 0.4), (16, 0.3) | 33, 105, 117 |

Each cell runs at the budget where random sets average 50% in that model's regime-map stage 1. Where the curve never reaches 50%, it runs at the budget closest to it. Budgets are frozen in `budgets.json` the first time they are read, so Chung–Lu and PAPER need their stage 1 first.

**Seed sets per pair.** There are 3 new pairs per cell and about 480 sets per pair; no set uses a test vertex.
- Random: 350 uniform random sets, the first sets of a 20,000-set pool.
- Degree-tilted: 10 sets for each γ in −1, −0.5, 0.5, 1, 2 and 3, with vertices drawn with probability ∝ degree^γ.
- Quantile picks: for each of degree, harmonic, pair mass, uniqueness and the frozen score, the pool set at each of 12 quantiles, from lowest to highest.
- Round 1's rules, best of 100 and of 1,000 by the frozen score, and best of 1,000 by degree.

Every set has:
- the core features, raw and as z-scores against the pool;
- the frozen score and its percentile in the pool;
- the 11 fixed-pair features;
- "oracle" features from B: how many of the set's edges and seed-to-non-seed clues survive in B, which no rule can see.

**Runs.** FAQ runs every set 3 times with different solver seeds; GOAT and percolation run each set once. The first pair of every cell has two extras:
- 100 random sets (10 per tenth of the score) on 10 fresh draws of B given A, as in `b_replicates.py`;
- for GOAT, the same sets with the regulariser halved and doubled.

**Size.** About 31,000 runs per model, roughly 5.5 hours with 3 jobs at the ER map's run times.

```bash
python -m selection_rule_comparison.landscape --model sparse_er
python -m selection_rule_comparison.landscape --model chung_lu --status
python -m selection_rule_comparison.landscape --model sparse_er --summary      # first table per cell
python -m selection_rule_comparison.landscape --model paper_pa --check-pairs   # can every cell draw pairs?
python -m selection_rule_comparison.landscape --model sparse_er --levels 0.2 0.8 --pairs 1 --no-extras
python -m selection_rule_comparison.landscape --model sparse_er --cells faq:16:0.2 goat:16:0.2
```

Results go to `data/landscape/<model>/`, and reruns resume:
- `settings.json` and `budgets.json` freeze the design and the budgets;
- `d16_rho0.30/<algorithm>/k143/pair_XXX/` holds `sets.csv` (one row per seed set, with its design, seed vertices and features), `runs.csv` (one row per run), `redraws.csv` and `regularizer.csv` (pair 0), and `pair.json`.

`read(model)` joins runs with their sets' features; `read(model, "redraws")` and `read(model, "regularizer")` do the same for the extras.

**Analysis notebook (added 9 Nov).** The original sections of `landscape.ipynb` read the saved results for all three models and plot them, taking under a minute with the `ds` kernel. Section 3 plots the frozen score, then degree alone, against match ratio in each model's nine cells: grey points are random sets on the saved B, and blue points are the first pair's 100 extra sets, each averaged over the saved B and its 10 redraws. The other sections cover how spread out single runs are, how much of a run's outcome comes from the seed set rather than the draw of B, every set by design, building versus screening, degree tilt and feature quantiles, a feature heatmap, the 20% and 80% levels next to regime-map stage 2, the mean degree 16, ρ 0.2 cells, first-step accuracy as a cheap proxy, and GOAT's regulariser. Its plotting and table functions are in `landscape.py`.

**Separate single-B extension (11 Nov).** Section 3.1 adds wider-score figures while preserving the original grey/blue plots. `landscape_extremes.py` uses pair 0's saved A/B in each of the 27 standard conditions, with the original budget, frozen weights, pool normalization and 120 fixed test vertices. It adds exact top-ten total-degree/harmonic sets among the 480 eligible vertices, heuristic high-value sets for the other metrics, and score-stratified intermediate sets using the fresh-pair construction methods. The full graph remains unchanged; test vertices are excluded only from seed eligibility. Existing first-run outcomes and A-only constructed-set outcomes are reused, and missing sets receive one run on the original B. Adaptive learned-selector proposals are excluded. Data and PNG/PDF figures are saved separately under `data/landscape_extremes/`. The new code cell's `RUN_WIDE_COLLECTION` flag controls collection; all other notebook sections keep reading the original data.

```bash
python -m selection_rule_comparison.landscape_extremes --jobs 3
```

## Weekend queue (7 Nov)

`weekend_queue.py` runs the long steps one after another on the Mac mini, for up to 35 hours. Each step is an ordinary command that resumes where it stopped.

| # | Step | What | Rough hours |
|---|---|---|---|
| 1 | `first_look` | Real pairs (`real_data_experiments/first_look.py`): both worms and two Enron week pairs | 0.5 |
| 2 | `landscape_er` | Landscape, ER | 5.5 |
| 3–4 | `map_cl_stage1`, `map_paper_stage1` | Regime map stage 1, Chung–Lu and PAPER: they set those models' landscape budgets | 1 + 1 |
| 5–6 | `landscape_cl`, `landscape_paper` | Landscape, Chung–Lu and PAPER | 5.5 + 5.5 |
| 7–8 | `map_cl_stage2`, `map_paper_stage2` | Regime map stage 2, Chung–Lu and PAPER | 1.5 + 1.5 |
| 9 | `fillers_levels` | Landscape, ER, the same cells at the 20% and 80% levels (1 pair, no extras) | 2.5 |
| 10 | `fillers_cells` | Landscape, ER, mean degree 16 and ρ 0.2 for all three algorithms (300 seeds) | 1.5 |
| 11 | `first_look_wiki` | Real pairs: Wikipedia (1,382 vertices, slow) | 3 |
| 12 | `more_pairs` | Landscape: a 4th and then a 5th pair per cell, on all three models | until the deadline |

At the deadline the running step stops cleanly and keeps its finished runs; the remaining steps are skipped. A failed step is logged and the queue goes on.

```bash
python -m selection_rule_comparison.weekend_queue --smoke     # about 10 minutes; results in tmp/weekend_smoke/
python -m selection_rule_comparison.weekend_queue             # the queue (default deadline 35 hours)
python -m selection_rule_comparison.weekend_queue --status    # from another window
python -m selection_rule_comparison.weekend_queue --stop      # stop it cleanly
python -m selection_rule_comparison.weekend_queue --steps landscape_cl landscape_paper --hours 12
python -m selection_rule_comparison.weekend_queue --dry-run   # print the plan and the commands
```

Logs go to `logs/weekend/<start time>/`: `queue.log`, `state.json` and one log per step.

## Learned selector (10 Nov)

`learned_selector.py` asks whether, at each landscape cell's budget k, a seed-set score fitted where good sets live picks sets that beat the best hand-made rule. The frozen pooled score was fitted at round 1's settings on sets from a random pool, with single-run pass/fail labels. Rule-built sets sit far outside that range: top degree is about 11–13 SD above random sets in seed degree, and the farthest training sets reach about 5 on ER. `learned_selector.ipynb` reads the results; its first section needs no new runs.

**Steps.**
1. **weights** (no runs). Per standard cell, test accuracy on the four core z-scores over random sets, and whether a score fitted in one cell ranks the redrawn sets of another (`weight_table`, `transfer_matrix`).
2. **constructed.** On the first landscape pair of each of the 27 standard cells, about 90–95 sets that span random to rule-built:
   - round 1's rules and the best-of screens;
   - degree tilts: 3 landscape sets per γ, plus 5 new sets at γ 5 and 8;
   - quantile picks near the top of each feature, and the bottom of degree and the score;
   - swap paths that turn 2 random sets into each rule's set, with sets at 25%, 50% and 75% of the way.

   Each set runs once on the pair's saved B and on the landscape's 10 redraws of B. With the landscape's 100 redrawn random sets, every set then has a label averaged over the same 11 Bs.
3. **round** (two rounds). Per cell:
   - fit the selector: logistic in the four core z-scores plus degree squared, fitted to mean accuracy over the 11 Bs;
   - search by single swaps from each rule's set and the best labelled sets, within the 2k highest-degree vertices plus the rules' vertices, keeping every z-score within the labelled sets' range (±0.5);
   - run 2 sets chosen by predicted accuracy and 2 by predicted accuracy plus its bootstrap SD on the same 11 Bs, then refit.
4. **finalize.** The final selector per cell (`final_model.json`, with leave-one-family-out checks).
5. **fresh.** 10 new pairs per cell at the same k. Each pair gets, on its saved B and 4 redraws:
   - the final selector's set;
   - the frozen score's set built by the same search;
   - every rule (the best rule is fixed in advance from the landscape, in `best_rules.json`);
   - the best of 1,000 random sets by the frozen score.

   Each pair also gets 20 random sets, run on its saved B only.

**Size.** About 27,500 runs for step 2, 2,400 for the rounds and 16,000 for the fresh pairs. That is roughly 5.5, 0.5 and 3.5 hours with 3 jobs at the landscape's run times.

```bash
python -m selection_rule_comparison.learned_selector --smoke            # every step, tiny; results in tmp/learned_selector_smoke/
python -m selection_rule_comparison.learned_selector --all              # every step in order (about 10 hours)
python -m selection_rule_comparison.learned_selector --step constructed
python -m selection_rule_comparison.learned_selector --step round --round 1
python -m selection_rule_comparison.learned_selector --step round --round 2
python -m selection_rule_comparison.learned_selector --step finalize
python -m selection_rule_comparison.learned_selector --step fresh
python -m selection_rule_comparison.learned_selector --status           # from another window
python -m selection_rule_comparison.learned_selector --step weights     # step 1's summary, no runs
```

Results go to `data/learned_selector/<model>/<cell>/<algorithm>/k<budget>/`, and every step resumes:
- `pair_000/constructed_sets.csv` and `constructed_runs.csv`;
- `pair_000/proposals.csv` and `proposal_runs.csv`;
- `model_round<r>.json` (the fit each round's proposals came from) and `final_model.json`;
- `fresh/pair_XXX/sets.csv` and `runs.csv`;
- `settings.json` and `best_rules.json` at the top.

The constructed sets reuse the landscape's pairs and redraws, which are checked by hash, so step 2 needs the landscape results in `data/landscape/`.

**Prediction plots (11 Nov).** Section 3.1 of `learned_selector.ipynb` shows the saved final selectors' predicted
accuracy against observed mean accuracy on their training sets (11 Bs), with a `y = x` reference. Section 4.1
compares learned predictions and frozen scores side by side for exactly the same evaluated strategy outputs on
the ten fresh pairs per setting. Selected sets have five Bs; grey random sets have only one B and are labelled
accordingly. These cells read saved features and models without refitting or running matching. The original
cut-degree plots remain in section 2. PNG and PDF copies go to `data/learned_selector/figures/`.

## Constructed sets on the redraws of B (11 Nov)

`constructed_redraws.py` (and `constructed_redraws.ipynb`) runs every constructed set of the fresh pairs on `b_replicates`' 10 draws of B. Each set then has a success rate over 11 Bs (the saved B is draw 0) with an interval, on the same Bs as the 100 random sets per pair. It also adds lowest-score sets and more intermediate sets.

**Sets, per pair.** All are chosen on A only, with the frozen score and the pair's original normalisation.
- Every distinct set in `selected_sets.csv` and `metric_sets.csv` (2,588 in all). Their saved-B runs are reused.
- The 3 lowest-scoring sets found by swap search, below the random sets.
- 40 new intermediate sets. Swap paths run one vertex at a time from 5 starts to 8 targets.
  - Starts: the random sets at the 5th, 35th, 65th and 95th score percentiles, and the top-degree rule set.
  - Targets: the best set of each metric group, and the lowest-score set.
  - Two sets are kept per equal-width score bin: 13 bins above the random range, 4 across it and 3 below it. Within a bin they are spread over targets.
  - They also run on the saved B.

**Draws of B.** `b_replicates`' draws, regenerated from its seeds. Each draw's hash is checked against its `draws.json`.

**Order.** Draw by draw: every set on every pair gets draw 1 before any set gets draw 2. A stopped run therefore leaves every set with the same number of draws.

**Running it.** 3,749 sets and 38,651 runs, about 5–6 hours with 3 jobs. Keep `--blas-threads 3`, as for the saved-B runs.

```bash
python -m selection_rule_comparison.constructed_redraws --jobs 3
python -m selection_rule_comparison.constructed_redraws --status      # from another window
python -m selection_rule_comparison.constructed_redraws --summary     # tables and figures only
```

**Results** go to `data/fresh_pair_scores/constructed_redraws/<algorithm>/<model>/pair_XXX/`:
- `catalog.csv`: one row per set (`set_id`), with its groups (`sources`), features, score and saved-B run. `catalog.json` records how the list was built.
- `draw_00.csv` (new sets on the saved B) to `draw_10.csv`: one row per run. `draw_XX.json` holds the draw's hash and its check against `b_replicates`.

**Summaries.**
- `set_rates()` gives each set's success rate and Wilson interval.
- `above_random_table()`, `membership_table()`, `group_table()` and `pair_table()` summarise the rates.
- `plot_overview()` and `plot_setting()` plot score against success rate, with the random sets alongside.
- `--summary` saves all of these to the results folder.

## Score-range banks: extra draws of B (21 Nov)

Adds draws 3–10 of B to the 27 synthetic banks of `data/score_range_collection/`, so every seed set has
11 draws and the weighted score can be plotted against each set's mean outcome across the full score
range. Details: `SCORE_RANGE_DRAWS.md`.

```bash
python -u -m selection_rule_comparison.score_range_draws --smoke --hours 0.2
caffeinate -i python -u -m selection_rule_comparison.score_range_draws --hours 10 --jobs 3
python -m selection_rule_comparison.score_range_draws --status      # from another window
```

**Results** go to `data/score_range_draws/<bank>/<algorithm>_draw_XX_runs.csv`. Analysis:
`score_range_11_draws.ipynb` (helpers in `score_range_draws_analysis.py`).
