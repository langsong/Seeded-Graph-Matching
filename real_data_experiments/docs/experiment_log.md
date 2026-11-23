> Dated log of the experiments in this folder, moved here from `README.md` on 23 Nov 2026. Paths and commands are as
> they were when each section was written; `../README.md` says where things are now (retired experiments are
> in `archive/`, notebooks in `notebooks/`).

# Real-data experiments

Real graph pairs for testing seed selection outside the synthetic models: the C. elegans connectome (hermaphrodite and male), English vs French Wikipedia, and Enron email weeks. The raw files come from the [SGM paper's data page](https://www.cis.jhu.edu/~parky/SGM/) and sit in the repository's `Data/` folder.

## Overnight confirmation and active experiments (18 Nov)

The updated `strategy_comparison.ipynb` and `active_test.ipynb` are **analysis only** and read
`results/overnight_v2/`. Collect the revised design separately:

```bash
python -u -m real_data_experiments.overnight --hours 6.5 --jobs 3
python -m real_data_experiments.overnight --status
```

See [OVERNIGHT.md](OVERNIGHT.md) for the design, restart instructions, output layout, and validation.
It includes independent budget calibration, repeated test partitions, corrected ensemble uncertainty,
paired active trajectories, and one-query diagnostics. Original saved results remain intact.

## Original repeated strategy comparison

`strategy_comparison.py` supplies the original collector and plotting functions for all 12 configured
native real pairs over the first-look budget grids. The notebook now displays the revised overnight
design above; the original data can still be read with `strategy_comparison.read()`.

Methods include exact top degree, randomized degree/clustered/spread/distinguishing/hub-free rules,
an A-only randomized adjacency-signature entropy rule, best-of-1,000 random sets by the frozen ER
score, and multi-start improving-swap `best_found` ER-score search. Each search includes its paired
best-of-n set. Random-reference normalization is fixed per graph/budget; no real outcomes enter selection.
Repeated draws preserve their selection frequencies, while duplicate seed sets share solver evaluations.
The notebook explains the randomized variants and the distinction between seed and solver variation.

Defaults: 20 policy draws on small pairs, 10 on Wikipedia, three FAQ/GOAT runs per distinct set,
GOAT regularizer 300, and one percolation run. The full experiment can take many hours.
Results are saved under `results/strategy_comparison/`; rerun to resume. A nine-hour terminal run is:

```bash
caffeinate -i python -u -m real_data_experiments.strategy_comparison --hours 9 --jobs 3
python -m real_data_experiments.strategy_comparison --status
```

The original CLI can restrict work with `--cases` and `--budgets`; these restrictions do not change the
saved experimental design. For an experiment with different repeats or settings, use a new
output directory. `python -m unittest real_data_experiments.test_strategy_comparison` checks score
updates, randomized set validity, search improvement, deduplication, resume and balanced aggregation.

## Quick start

The converted archives are already in `data/`. To rebuild them, for example after changing `Data/`, run from the repository root (about 2 seconds):

```bash
python -m real_data_experiments.prepare_data
```

Then, in any experiment:

```python
from real_data_experiments.datasets import get_pair

pair = get_pair("worm")                         # chemical vs electrical, 279 neurons
A, B, truth = pair.for_matching(shuffle_seed=0)
```

`for_matching` returns the same triple as `experiment_common.graphs.generate_graph_pair`: dense float adjacency matrices and `truth`, where `truth[v]` is the vertex of B that matches vertex v of A. So real pairs go straight into the existing runners, for example as in `selection_rule_comparison/real_graph_check.py`:

```python
import numpy as np
from experiment_common.sgm import run_sgm_once
from experiment_common.percolation import run_percolation_once
from selection_rule_comparison import settings as S

seeds = np.random.default_rng(1).choice(pair.n, size=50, replace=False)   # vertices of A
faq = run_sgm_once(A, B, truth, seeds, 0, S.SGM["max_iter"], S.SGM["tol"])
perc = run_percolation_once(A, B, truth, seeds, algorithm_seed=S.PERCOLATION["algorithm_seed"],
                            threshold=S.PERCOLATION["threshold"],
                            expand_when_stuck=S.PERCOLATION["expand_when_stuck"])
faq["final_unseeded_accuracy"], perc["final_unseeded_accuracy"]
```

Always match the shuffled copy from `for_matching`. A `RealPair` stores both graphs aligned (vertex i of A is vertex i of B), and with an unshuffled B, ties in the assignment step can land on the right answer by accident.

## Files

| File | Contents |
|---|---|
| `prepare_data.py` | Converts `Data/` into `data/`, checks the result against published numbers, writes `data/summary.csv`. `--faq-check` also reruns seeded FAQ on the worm and Wikipedia (about 1–2 minutes). |
| `datasets.py` | `get_pair` (build a pair), `RealPair.for_matching` (shuffle B for matching), `describe` / `summary_table` (sizes, degrees and agreement), `load_arrays` (the stored arrays). |
| `rdata_reader.py` | A small reader for R `save()` files, so the `.Rd` / `.Rdata` downloads need no R install. It raises an error on anything it does not understand. |
| `data/*.npz` | One archive per dataset (below). |
| `data/manifest.json` | SHA-256 hashes of the raw files, array shapes and dtypes, notes on the choices below. |
| `data/summary.csv` | One row per standard pair: sizes, degrees, isolated vertices, common edges, edge correlation. |

## Datasets

| `get_pair(name)` | Graphs (A vs B by default) | Vertices | Raw files |
|---|---|---|---|
| `"worm"` | C. elegans hermaphrodite: chemical vs electrical (gap-junction) synapses | 279 neurons | `elegansGraph.mat`, `herm_Graph.Rd` |
| `"worm_male"` | C. elegans male (posterior): chemical vs electrical | 167 neurons named in both | `elegansGraph_male.mat` |
| `"wikipedia"` | English vs French articles near "Algebraic Geometry", hyperlinks | 1,382 articles | `GE.Rdata`, `GF.Rdata` |
| `"enron"` | One week of email vs another (default weeks 130 and 131) | 184 addresses | `enron.mat` |

In every pair the true correspondence is the identity (row i is the same neuron, article or person in both graphs).

## Options

`get_pair(name, graph_a=None, weeks=(130, 131), weighted=False, directed=False, loops=False, vertices="all")`

- **`graph_a`**: which graph seeds are chosen from. `"chemical"` (default) or `"electrical"` for the worms, `"english"` (default) or `"french"` for Wikipedia. For Enron, A is the first of `weeks`.
- **`weeks`**: Enron only, numbered from 1 as in MATLAB and the SGM paper (which matched weeks 130, 131 and 132).
- **`weighted`**: keep synapse counts (worms). Otherwise every edge has weight 1.
- **`directed`**: keep directions (worm chemical synapses, Enron). Otherwise an edge in either direction becomes one undirected edge, with counts added if weighted.
- **`loops`**: keep self-loops (3 in the hermaphrodite's gap junctions, self-sent Enron email).
- **`vertices`**: `"all"`; `"nonisolated"` drops vertices with no edges in either graph (the 253-neuron worm of Patsolic et al.); `"common_lcc"` keeps the vertices in the largest connected component of both graphs. Both rules are re-applied until nothing more is removed, and `pair.settings["dropped"]` lists what was removed.

The defaults (unweighted, undirected, no self-loops, all vertices) are the SGM paper's setup. Isolated vertices cannot be matched from structure, so `"all"` caps accuracy below 1; use it to compare with published numbers and `"nonisolated"` or `"common_lcc"` when that cap would get in the way.

## What the pairs look like

From `data/summary.csv` (defaults unless stated). Edge correlation is the Pearson correlation of the two graphs' edge indicators over vertex pairs; for a correlated ER pair it equals the generator's ρ, so it places a real pair roughly on the regime map's ρ axis. Kept share is the share of A's edges also present in B.

| Pair | n | Mean degree A / B | Degree CV A / B | Isolated A / B | Edge correlation | Kept share |
|---|---|---|---|---|---|---|
| Worm, all | 279 | 14.1 / 3.7 | 0.80 / 1.17 | 0 / 26 | 0.17 | 0.10 |
| Worm, common LCC | 247 | 13.1 / 4.1 | 0.84 / 1.07 | 0 / 0 | 0.18 | 0.12 |
| Worm male | 167 | 20.2 / 10.9 | 0.63 / 0.65 | 1 / 2 | 0.37 | 0.31 |
| Wikipedia, all | 1,382 | 27.3 / 21.7 | 1.30 / 1.47 | 0 / 59 | 0.46 | 0.42 |
| Enron 130 vs 131 | 184 | 4.1 / 4.1 | 1.64 / 1.63 | 38 / 33 | 0.87 | 0.88 |
| Enron 130 vs 132 | 184 | 4.1 / 4.2 | 1.64 / 1.60 | 38 / 30 | 0.74 | 0.75 |

The worm's two graphs barely agree: only 188 of the chemical graph's 1,961 undirected edges are also gap junctions, below the lowest ρ (0.2) in the synthetic regime map. Wikipedia sits in the middle of the map's ρ range; Enron's consecutive weeks are near the top, but the graphs are sparse and very uneven.

## Checks

`prepare_data.py` checks the converted data against published numbers; all pass:

- Worm: 279 neurons; 2,194 directed chemical edges; 514 gap-junction edges with 26 neurons that have none (Chen et al. 2016); electrical largest component 248 (SGM paper); 253 neurons without isolates (Patsolic et al.).
- Wikipedia: 1,382 articles; the English graph is connected; the French largest component has 1,323 articles (SGM paper).
- Enron: weeks 130 and 131 have 488 and 482 directed edges including self-loops and largest components of 146 and 151, the pair iGraphMatch ships as `Enron`.

`--faq-check` reruns seeded FAQ (scipy's implementation, uniformly random seeds, 5 runs each) and reproduces the SGM paper:

| Pair | Seeds | Mean accuracy on non-seeds | SGM paper |
|---|---|---|---|
| Worm, all 279 | 200 | 0.157 (0.114–0.215) | about 16% |
| Wikipedia, all 1,382 | 50 | 0.562 (0.549–0.592) | above 50% |

## Notes on the raw files

- **Worm, hermaphrodite.** `elegansGraph.mat` and `herm_Graph.Rd` list the same neurons in the same order with the same edges. Synapse counts come from the MATLAB file (whole numbers); the R file's chemical weights differ on 1,584 of 2,194 edges (some are halved), so weighted analyses should say which file they use. `pair.vertex_class` decodes the R file's `vcols` as motor (118), interneuron (83) or sensory (78), following the SGM data page. The R file's `types` codes and 3-column `Q` table are kept in the archive unlabelled.
- **Worm, male.** The chemical graph has 170 neurons and the electrical graph 167. The pair keeps neurons named in both after removing `*` from names, which treats `AN1b*` and `*AN1b` as one neuron; ALNL, ALNR and PLNL have no gap-junction row and are left out. 81 male neuron names also appear in the hermaphrodite, if a cross-sex comparison is ever wanted.
- **Wikipedia.** `GE.Rdata` and `GF.Rdata` hold shortest-path distances, not adjacency, so an edge is a distance of 1. `GF` codes some long or unreachable distances as 6, which does not affect the edges.
- **Enron.** `enron.mat` is a 184 × 184 × 187 array of directed weekly graphs, with entry `[i, j, t]` used as stored. It is the same file as `utils/enron.mat`, which `real_graph_check.py` uses to build its edge-subsampled pairs.

## First look (7 Nov)

`first_look.py` asks whether round 1's rules and the frozen scores pick better seeds than random sets on the real pairs.

**Pairs.** All unweighted and undirected; A is the graph seeds are chosen from.
- `worm`: all 279 neurons.
- `worm_lcc`: the 247 neurons in the common largest component.
- `worm_male`: 167 neurons.
- `enron_130_131` and `enron_130_132`: the common largest component of each pair of weeks, 145 and 141 addresses.
- `wikipedia`: all 1,382 articles. It is much slower, so it runs on its own, with fewer random sets and one FAQ run per set.

**Design.** 20% of each pair's vertices are test vertices, never seeds, and B's labels are shuffled once. At each budget the pair gets:
- 30 random sets (15 for Wikipedia) from a pool of 1,000;
- round 1's six rules;
- the best set of the pool by each model's frozen score (ER, Chung–Lu, PAPER weights) and by degree.

Every set gets the core features with z-scores against the pool, and all three models' scores with their percentiles.

**Variants.** Percolation, FAQ (3 runs per set), and GOAT with regulariser 100 and 300.

**Outcome.** Accuracy on the test vertices. For the worm it also records accuracy by neuron class and `same_class_test`, the share of test neurons matched to a neuron of the same class (chance is about 0.35).

```bash
python -m real_data_experiments.first_look                     # the five small pairs, about half an hour
python -m real_data_experiments.first_look --pairs wikipedia    # a few hours
python -m real_data_experiments.first_look --status
python -m real_data_experiments.first_look --summary            # random sets vs rules and screens, per budget
```

Results go to `results/first_look/<pair>/<variant>/k<budget>.csv` (one row per run, with the set's features and seed vertices) and `results/first_look/<pair>/pair.json`. `read()` loads everything, and reruns resume.

**Analysis notebook (added 9 Nov).** `first_look.ipynb` reads these results and plots them. It runs no matching and takes under a minute with the `ds` kernel. It shows accuracy against budget for each variant, each rule's and screen's gain over random sets, the frozen scores against accuracy, the four variants side by side, and the worm by neuron class. Its plotting and table functions are in `first_look.py`.


## Follow-up: wider candidates and saved synthetic selectors (11 Nov)

Sections 7–11 of `first_look.ipynb` add an executable follow-up in `follow_up.py`. The original first-look data
and plots remain the baseline. No model is trained on real data.

- **Saved-model transfer.** The existing sets are scored with all nine saved source settings per algorithm.
  GOAT uses compatible source regularizers: three ER models at r=100 or six CL/PAPER models at r=300. The fixed
  transfer policy averages compatible predictions equally; source models are not chosen using real outcomes.
- **Solver checks.** FAQ at 100 vs 500 iterations, and GOAT at 100 vs 300, with common solver seeds and three runs
  per unique set. Warnings and convergence are saved. The earlier Enron 100–101/102 window is a development
  diagnostic; the already-inspected 130–131 pair is an exploratory iteration-sensitivity check.
- **Wider candidates.** Exact top ten eligible sets by total degree/harmonic, ten best encountered sets for cut
  degree, pair mass, uniqueness, coverage and each of the three frozen scores, plus intermediate swap paths.
  Random sets, rules and pool screens are retained. Every model reranks the same deduplicated candidate bank.
  Each individual synthetic source ranks inside its original feature-range guard where possible, with an
  out-of-range fallback if no candidate qualifies. The equal-weight mean ranks the full bank and reports range
  coverage. All feature normalization uses a fixed 1,000-set pool for that graph/test split/budget.
- **Native replication.** Three test-split/relabeling repetitions by default; these do not create new natural
  graph pairs. New Enron tests use weeks 145–146/147 and 160–161/162. Shared first weeks define shared time
  windows and must not be counted as independent replicates.
- **Separate controls.** Real-topology simulations fix A, test vertices and candidate sets, and redraw B with
  copy probability 0.8 plus independent background edges at A's density. Five B draws by default. These are
  reported separately from the natural pairs.

The notebook's collection flags default to False; it opens as analysis of saved results. A completed initial
subset covers three Enron pairs at k=10 with two test splits (FAQ), an iteration/regularizer diagnostic subset,
and a three-B Enron control. Wikipedia's wider FAQ candidate banks at k=10/20/30 are prepared; their new matching
runs are pending. The existing Wikipedia outcomes already support the transfer plots in section 7.

```bash
# Resume the first expanded Enron subset, or change the case/budget/repetition arguments:
python -m real_data_experiments.follow_up --phase wide --cases enron_130_132 enron_145_147 enron_160_162 --variants faq --budgets 10 --replicates 2
# Inspect candidate plans without matching (useful before the slower Wikipedia runs):
python -m real_data_experiments.follow_up --phase wide --cases wikipedia --variants faq --budgets 10 20 30 --replicates 1 --plan-only
# Collect prepared Wikipedia candidates, or run the full configured phase:
python -m real_data_experiments.follow_up --phase wide --cases wikipedia --variants faq --budgets 10 20 30 --replicates 1
python -m real_data_experiments.follow_up --phase stability
python -m real_data_experiments.follow_up --phase controls
python -m real_data_experiments.follow_up --phase wide --status
```

New files live under `results/follow_up/<phase>/<case>/<variant>/k###/rep_###/`: `sets.csv`, `memberships.csv`,
`predictions.csv`, `runs.csv`, and `pair.json`. The top-level settings freeze solver/search settings, weights
and saved-model hashes; the source manifest records the actual models. Unique sets share outcomes across policy
memberships. Incomplete candidate-bank repetitions are excluded from performance comparisons until all sets
have the same solver-run budget. Run only one collector for a given task at a time. Figures are saved to
`results/follow_up/figures/`; changing the collection design requires a new `REAL_FOLLOW_UP_OUTPUT_DIR`.

## Two-phase active querying: initial test (17 Nov)

The original `active_test.py` pilot asks whether, at the same total budget K, spending the second half
of the queries on the vertices the matcher is least sure about beats spending them on more structural seeds
(the approach of Malmi, Gionis & Terzi 2017).

- **Policies.**
  - Static: random (10 draws), top degree, clustered, distinguishing and hub-free distinguishing.
  - Two-phase: the first K/2 seeds by top degree, then batches of 5 (Enron) or 10 (Wikipedia), re-running
    FAQ or GOAT after each batch. Batches are chosen by matcher confidence (GOAT's soft row maximum; FAQ's
    agreement across 3 restarts), by edge consistency, or at random as a control (5 draws).
  - After each re-run, the corrected previous matching is kept if it preserves more edges (no truth used).
- **Outcome.** Every vertex can be queried. The main outcome is correct matches as a share of n, seeds
  included (`acc_all`). Per round, `rounds.csv` records each certainty's AUC for the current errors and
  the share of the queried batch that was wrong (`hit_rate`).
- **Defaults.**
  - Enron 130–132, 145–147 and 160–162 (common LCC) at K = 10, 20, 30.
  - Wikipedia at K = 20 and 40 on request.
  - FAQ: 3 restarts (1 on Wikipedia). GOAT: regularizer 300.

```bash
python -m unittest real_data_experiments.test_active_test        # bookkeeping checks, no graspologic needed
python -m real_data_experiments.active_test --smoke              # under a minute, results/active_test_smoke/
python -m real_data_experiments.active_test                      # three Enron pairs, a few minutes
python -m real_data_experiments.active_test --cases wikipedia    # about 30-60 minutes
python -m real_data_experiments.active_test --summary
```

The notebook now reads the revised overnight design described at the top of this file. The original
pilot's results remain in `results/active_test/` (`settings.json`, `finals.csv`, `rounds.csv`). Finished blocks are
skipped on rerun. A changed design needs a new folder (`--output-dir`).
