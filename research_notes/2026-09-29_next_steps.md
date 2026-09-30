# Next steps: seed-set quality experiments

*Plan as of 29 Sep 2026. Covers direction 1 (can we tell good seed sets from bad ones, and which features matter) and sets up direction 3 (greedy selection by marginal gain).*

## Decisions so far

- **Graph models.** Sparse correlated ER is the homogeneous control, and PAPER (linear preferential attachment tree + ER noise, shared tree between A and B) is the primary heterogeneous model. Both run at mean degree about 5–6, n = 600.
  - Later: vary heterogeneity inside PAPER with mean degree fixed (uniform attachment → APA(2,1) → linear preferential attachment).
  - Chung–Lu stays only as a degree-matched control for PAPER, after the generator fix.
  - IER with independent Beta edge probabilities is dropped: it is exactly correlated ER.
  - SBM comes later, as a community-structure axis.
- **Outcome.**
  - Run SGM for up to 150 iterations and record accuracy and objective score at iterations 10, 20, 30, 50, 100 and 150.
  - Success = at least 90% accuracy on non-seed vertices.
  - The label is the seed set's success probability p(S), estimated from 5 runs on evaluation sets.
  - Training sets get 1 run each.
  - Best-of-5 (by objective) is recorded as a separate outcome, never as the label.
- **Design.** Several pairs, analysed within pair: features and outcomes are compared only between seed sets on the same pair. A single fixed pair is kept for piloting and mechanism work.
- **Features.** A compact panel, one feature per aspect (step 4).
- **Seed-set sampling.** Random sets, plus wide-range sets from degree-weighted sampling and swap paths away from constructed sets. Test sets are random + constructed.

## Steps

### 0. Housekeeping and code structure
- Snapshot commit and push (done 29 Sep).
- Tag `main` as `reu-summer-2026`.
- Move legacy summer code into `legacy_reu/` with `git mv`: `Experiments.py`, `experiment_sweep.py`, `test_seeding.py`, `tests.py`, `results/`, and the first-direction notebooks.
- Pull the shared code out of `cross_pair_seed_set_quality/data_collection.py`, `utils/` and `ExpandWhenStuck.py` into one package: `graphs.py`, `matching.py`, `features.py`, `presets.py`, `io.py`. Experiment folders then only hold scripts, notebooks and data.
- Remove the import-time `np.random.seed(1234)` in `config.py`, and seed explicitly per pair and per run.
- Generator fixes:
  - Chung–Lu: fixed quantile weights, capped at √Σw, instead of the global rescale.
  - Rename the `paper` key to `pareto_cl`, keeping the old generator for reproducing saved data.
  - Allow β = 0 (uniform attachment) in PAPER.
  - Shuffle Graph A's labels in PAPER, so a vertex's index doesn't reveal its age.
  - Remove IER.
- Pin `requirements.txt` (Python and graspologic versions). Add `tests/` with known-answer tests.
- **Done when:** re-evaluating a few saved seed sets per fixed pair reproduces the saved features and SGM outcomes exactly, and the tests pass.

### 1. Model exploration notebook
- For ER, PAPER (three attachment settings) and the fixed Chung–Lu model:
  - the degree distribution on log-log axes, and the max degree;
  - the fraction of A's edges that also appear in B, and common edges per vertex;
  - the fraction of vertices with 0 or 1 common edges (a rough ceiling on matchable vertices);
  - structurally identical vertices;
  - for PAPER, the leaf fraction and sibling-group sizes.
- **Done when:** there is a one-paragraph takeaway per model.

### 2. Algorithm review and protocol
- **SGM:** 150 iterations, with the checkpoints above. Don't use the `tol`-based "converged" flag as a success signal. Give every run its own recorded RNG seed.
- **Percolation (ExpandWhenStuck):**
  - the implementation follows Kazemi et al.; seed its tie-breaking per run;
  - decide whether the success label is recall (current), precision or both, and keep it consistent with SGM;
  - add known-answer tests: identical graphs, tiny hand-built graphs, and a sharp success jump as the seed count grows on correlated ER.
- **Done when:** both algorithms pass the tests and share one outcome schema.

### 3. Calibration sweep
- ER and PAPER, separately for each algorithm, on 3 pairs per model:
  - a coarse screen over k (10–20 random sets × 1 run per k);
  - then a confirmation at the chosen k (50 sets × 3 runs).
- Target:
  - random-set success of 40–60%;
  - top-degree clearly better than random, so there is room for a better rule;
  - differences between seed sets not swamped by solver noise.
- Also record the smaller k at which top degree succeeds 50% of the time, for later rule comparisons.
- **Done when:** k is fixed per model and algorithm and written into the presets.

### 4. Features and seed-set generator
- The panel:
  - mean seed degree (sum of log degrees on PAPER);
  - fraction of non-seeds with at least two seed neighbours;
  - mean 3-hop seed count (compactness);
  - seed-neighbourhood Jaccard (redundancy);
  - 3-hop resolving fraction (distinguishability; needs a 1–2 hop version in dense graphs);
  - number of seeds with degree ≤ 2.
- Store seed lists and outcomes; compute features in a separate pass.
- Freeze the panel before the main analysis.
- Replace the farthest-point "feature-diverse" sampler with degree-weighted sampling (P(v) ∝ deg^γ, γ from −1 to 3) plus swap paths from constructed sets (top-degree, greedy coverage, greedy resolving, spread-out).

### 5. Pilot on one fixed pair
- About 200 seed sets end to end, to debug the pipeline and check label balance and run-to-run reliability.

### 6. Main experiment
- Per model and algorithm: 10 pairs × 100 training sets × 1 run, plus 20 test sets per pair × 5 runs.
- Analysis within pairs: within-pair AUC, leave-one-pair-out transfer, and a comparison with the noise ceiling.

### 7. Feature effects by intervention
- Swap one seed to raise a single feature while roughly holding the others fixed, and measure the change in p(S). This separates correlated features, which regression importance can't do, and gives the first marginal-gain measurements for direction 3.

## Proposed success criteria (to agree before step 6)
- Within-pair AUC ≥ 0.75 against the 5-run labels.
- The top 10% of seed sets by predicted score beat random sets by a margin the group sets in advance.
- Both measured against a baseline model that uses mean seed degree alone.

## Compute
About 2,000 SGM runs per model for step 6, plus a few hundred for calibration, and similar numbers for percolation. That is a few CPU-hours per model when run in parallel.

## Open questions for the group
- PAPER pairs: keep the shared tree, or also test a version where every edge is noisy (subsampling one parent graph)?
- Percolation label: recall, precision, or both?
- Keep a fixed 90% success threshold, or measure accuracy relative to what is achievable on each pair?
- Will this branch ever merge back into `main`?

## Evidence behind these decisions
From exploratory reruns, 28–29 Sep. Most reruns used the `sgmlib` re-implementation of graspologic's SGM, validated at 30 iterations (0.52 vs 0.54 success on the same sets).
- **Iteration cap:** 977 of 1,000 saved `paper_pa` runs stopped at the 30-iteration cap without converging. Success rose from 0.52 at 30 iterations to 0.83 at 100 or more. 65% of runs failing at 30 succeeded later, and none went the other way.
- **Solver noise:** on `paper_pa`, differences between seed sets had about 0.4× the variance of run-to-run solver noise. 43% of sets had mixed outcomes over 5 runs. Predicting a single run is capped at about AUC 0.74.
- **Label-free signal:** the objective score separated successes from failures perfectly (AUC 1.00). Accuracy at iteration 10 reached AUC 0.95, versus 0.76 for H0.
- **Feature redundancy:** 7 of the 17 current features correlate at |ρ| ≥ 0.83 with mean seed degree, and 2 are constant on PAPER. The 3-hop resolving fraction was the best or tied-best single feature on all five fixed pairs.
- **Budgets too generous:** at the current k with 150 iterations, the top-degree set succeeded in every run on every fixed pair.
- **Cross-pair pooling:** on Pareto–CL the pooled AUC (0.89) came almost entirely from pair difficulty (0.88 from pair difficulty alone). Within pairs it was 0.71.
- **Pareto–CL all-fail pairs:** these are exactly the pairs where the generator's global rescaling fired, which happens in about 30% of draws at α = 2.
- **IER:** independent Beta edge probabilities give exactly correlated ER, with ρ_eff = ρ + (1 − ρ)/(α + β + 1).
- **Density:** at fixed correlation, denser graphs move the transition to fewer seeds at about the same coverage (about 15% of non-seeds with a seed neighbour).
- **Wide-range sampling:** the feature-diverse sets widened feature spread 1.3–1.9×. Top-degree sets sit 10–16 random-set standard deviations away, and adding the diverse sets did not improve prediction on random sets.
