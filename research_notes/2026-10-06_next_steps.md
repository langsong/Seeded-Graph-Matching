# Next steps (6 Oct 2026)

*Replaces the 29 Sep plan. Based on the 400-set pilots (ER, PAPER, Chung–Lu) and the first percolation checks.*

## Decisions
- **Fixed pairs stay the focus.** Analyse within each pair, and use 3 pairs per model. Which graph you draw explains 13–17% of SGM's outcome, so one pair can be atypical.
- **SGM target:** the number of successful runs out of 5, fitted with a binomial regression to estimate single-run success probability p(S). Evaluate probability prediction and ranking without converting the count to a majority label.
- **Practical SGM outcome:** measure whether the run with the highest objective among R restarts succeeds, directly from the saved runs. Under independent restarts, 1 - (1 - p(S))^R predicts the chance that at least one run succeeds. It does not by itself predict best-objective success. Report how often objective selection misses an available success, with ties resolved by run order.
- **Percolation target on sparse pairs:** mean non-seed accuracy over the runs. Record correct, unmatched, and incorrectly matched non-seeds using the same denominator. Low common-edge degree is a diagnostic of difficulty; it is not an established accuracy ceiling for ExpandWhenStuck, which also propagates support from tentative pairs.
- **Cross-pair work:** fit a separate model on each of 2 training pairs, then average their predicted probabilities on the 3rd and evaluate the resulting ranking. Each model uses its own training-fitted preprocessing. Choose regularization within each training pair; do not tune on the held-out pair. Report all 3 held-out results separately, with probability calibration treated as an additional transfer diagnostic.
- **Use the calibrated budgets below for the next collection.** SGM is calibrated against best-objective-of-5 success on random sets; percolation is calibrated against one-run success. Pair-specific rates can still vary substantially, so retain and report every graph-pair identity.

## Current settings (n = 600, 100 SGM iterations, 5 SGM runs per set)

| Model | Parameters | Seeds (k) |
|---|---|---|
| ER | p = 0.01, ρ = 0.7 | 26 |
| PAPER | background p = 0.005, ρ = 0.62 | 13 |
| Chung–Lu | α = 2, weight scale = 1.9260087733615983, no floor, ρ = 0.7 | 17 |

The pooled calibration estimates for best-objective-of-5 success were 55.6% for ER, 53.3% for PAPER, and 56.25% for Chung–Lu. These are tuning estimates rather than final performance results.

## Percolation settings (n = 600, ExpandWhenStuck, r = 2, 1 run per set)

| Model | Parameters | Seeds (k) |
|---|---|---|
| ER | p = 0.020, ρ = 0.7 | 8 |
| PAPER | background p = 0.0167, ρ = 0.62 | 7 |
| Chung–Lu | α = 2, weight scale = 4.076194702720809, no floor, ρ = 0.7 | 5 |

The ER and PAPER scans put their 50% transitions at about k=8.2 and k=6.5. For the updated floor-free Chung–Lu generator, k=5 succeeded for 27/60 random sets across three calibration pairs; k=6 succeeded for 34/60.

## Steps
1. **Code fixes before any new data.**
   - Record thread count and library versions.
   - Break ties at random in the selection rules.
   - Flush output files after every seed set.
   - Match degree-controlled contrasts on seed–non-seed degree, and place them in the middle degree bins.
   - Switch the notebook to grouped binomial regression: degree only, signatures only, degree plus signatures, and the full feature panel. Cross-validate on random training sets, keeping the five runs of each set together. The other training sets can remain in each training fold; the random test sets stay held out.
   - Select regularization and the primary ranking model using training cross-validation. Report held-out binomial negative log likelihood, per-run Brier score, calibration, and correlation with observed success fractions. Compare models on the same sets. Five-run fractions remain noisy estimates of p(S).
   - Compute best-objective accuracy, best-objective success, any-run success, and selection misses from the run-level CSV. Do not threshold the mean accuracy as a substitute for a successful-run count.
   - *Done when:* the updated notebook runs from top to bottom on the existing data and the user verifies that re-running a few saved seed sets reproduces their features and outcomes.
2. **Fixed-pair percolation data collection.** Start with one calibrated reference pair per model and collect 1,000 sets: 400 random training, 200 random test, 200 degree-weighted, and 200 degree-controlled contrast sets. Run one tie-break realization per set.
   - Analyse each reference pair separately. With one run per set, tie-break variance is intentionally left unestimated.
   - After the analysis is verified, repeat the same collection and analysis on two additional fixed pairs per model to assess graph-pair variability; do not pool the pairs automatically.
   - *Done for the initial diagnostic when:* there is one 1,000-set dataset per selected graph model.
3. **SGM on 2 more pairs per model.** Use 500 sets per pair: 200 random training, 100 random test, 100 degree-weighted, and 100 degree-controlled contrast sets. Run SGM 5 times for every set, including the test sets. Aim for about 40 signature contrast pairs per model, to test the candidate effect beyond degree seen in the pilot. Roughly 1–2 hours per pair.
   - Store an explicit contrast-pair ID. Match seed–non-seed degree, use middle degree bins, and use common solver RNG seeds for both sets in a contrast. Record remaining differences in coverage and centrality.
   - Report the high-minus-low success-fraction difference for every contrast, including ties, and summarize magnitudes by graph pair. A contrast controls degree but may still change other structural properties.
   - Reserve random test sets before feature or outcome analysis. Keep constructed sets and their contrasts in training. Three graph pairs per model are a replication step, not a precise estimate of transfer performance.
   - *Done when:* there are 3 pairs per model.
4. **Leave-one-pair-out test.** Use the separate-model averaging procedure above to rank the reserved random sets on the 3rd pair. Apply the same procedure to degree-only, signature-only, degree-plus-signature, and full-panel models. Freeze the comparison before inspecting the held-out outcomes.
   - *Done when:* each of the 3 held-out pairs has a ranking comparison against degree alone, including the number of evaluated sets. Interpret variation across these 3 results explicitly.
5. **Compare selection rules after collection (direction 3).** Use the frozen SGM budgets above and compare top-degree, greedy signatures, and random sets with equal seed budgets and restart counts; randomize rule ties and record their RNG seeds. The calibrated target is best-objective-of-5 success near 50% for random sets. Keep the single-run counts as the modelling outcome and report graph-pair-specific results.

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
