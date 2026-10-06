# Next steps (6 Oct 2026)

*Replaces the 29 Sep plan. Based on the 400-set pilots (ER, PAPER, Chung–Lu) and the first percolation checks.*

## Decisions
- **Fixed pairs stay the focus.** Analyse within each pair, and use 3 pairs per model. Which graph you draw explains 13–17% of SGM's outcome, so one pair can be atypical.
- **SGM target:** the number of successful runs out of 5, fitted with a binomial regression. "Success with 5 restarts" (the best-objective run succeeds) is computed from this model and is used later for headline comparisons.
- **Percolation target on sparse pairs:** mean accuracy over the runs. 90% is unreachable there, because about 10% of vertices have fewer than 2 edges present in both graphs. Record unmatched and wrongly matched vertices separately.
- **Cross-pair work:** train the within-pair model on 2 pairs and test how well it ranks seed sets on the 3rd. No models pooled across pairs.

## Current settings (n = 600, 100 SGM iterations, 5 SGM runs per set)

| Model | Parameters | Seeds (k) |
|---|---|---|
| ER | p = 0.01, ρ = 0.7 | 30 |
| PAPER | background p = 0.005, ρ = 0.62 | 16 |
| Chung–Lu | α = 2, floor 0.005, ρ = 0.7 | 22 |

## Steps
1. **Code fixes before any new data.**
   - Record thread count and library versions.
   - Break ties at random in the selection rules.
   - Flush output files after every seed set.
   - Match degree-controlled contrasts on seed–non-seed degree, and place them in the middle degree bins.
   - Switch the notebook to the binomial model with cross-validation over random sets.
   - *Done when:* re-running a few saved seed sets reproduces their features and outcomes.
2. **Percolation on the three pilot pairs.** Same 400 seed sets, 3 runs each (about 7 min per pair).
   - Compare with SGM on the same seed sets: do the sets that are good for one work for the other?
   - *Done when:* there is a fixed-pair percolation dataset and an SGM vs percolation comparison per model.
3. **SGM on 2 more pairs per model.** Current settings, 400 sets each, with the revised contrasts. Aim for about 40 signature contrast pairs per model, to test the one effect beyond degree seen on every pair so far. Roughly 1–2 hours per pair.
   - *Done when:* there are 3 pairs per model.
4. **Leave-one-pair-out test.** Train on 2 pairs, then rank seed sets on the 3rd. Compare against a degree-only model.
   - *Done when:* we know whether the learned ranking transfers to a new graph.
5. **Smaller budgets for comparing selection rules (direction 3).** First a short check that 5-restart success is about 50% for random sets: try about 25 seeds for ER, 12–13 for PAPER, and keep 22 for Chung–Lu. Then compare top-degree, greedy signatures and random at those budgets.

## Later / optional
- A dense setting where both algorithms are near 50% at the same budget: ER with average degree 24, about 9–10 seeds. Run noise is large there, so use 5 runs per set.
- Percolation with threshold r = 3 in dense graphs; r = 2 lets wrong pairs through more easily as density grows.
- H0 (first-step accuracy) as a cheap stand-in label for the greedy experiments (direction 1b → 3).

## Why (pilot evidence)
- On random seed sets, degree is the baseline everywhere, and the 9-feature model never beats the best single feature.
- Distance-signature contrasts: the high-signature set won 11 of 12 non-tied pairs across the three models.
- Solver noise: the seed set explains only 22–42% of one SGM run, but 94–98% of one percolation run on sparse pairs.
- At the current budgets, top-degree almost always succeeds, so selection rules can only be compared at smaller budgets.
- Details: the project docs `pilot-findings-three-models-2026-10-05.md` and `percolation-progress-review-2026-10-05.md`.
