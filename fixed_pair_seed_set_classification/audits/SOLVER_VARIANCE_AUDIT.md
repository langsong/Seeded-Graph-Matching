# Historical `paper` (Pareto–Chung–Lu) solver-variance audit

This audit holds one saved graph pair from the historical `paper` dataset (Pareto–Chung–Lu, not the newer `paper_pa` model) and each seed set fixed while changing only Graspologic's solver RNG seed. It samples 100 random and 100 feature-diverse seed sets from the existing training data, with five SGM runs per set (`n_init=1`, `max_iter=30`, `tol=0.01`). The default initialization is unperturbed; the RNG changes assignment input shuffling, which affects tie breaking. Success means at least 90% final accuracy on nonseed vertices. The run with RNG seed 2026 exactly reproduces each original saved outcome.

| Measure | Result |
| --- | ---: |
| Seed sets / SGM runs | 200 / 1,000 |
| Sets with both success and failure across five runs | 102 (51%) |
| Sets successful in zero / five runs | 53 / 45 |
| Original single-run label disagreeing with five-run majority | 39 (19.5%) |
| Median within-set range of final accuracy | 0.328 |
| Estimated reliability of one final-accuracy run / five-run mean | 0.60 / 0.88 |
| Estimated reliability of one H0 run / five-run mean | 0.76 / 0.94 |
| Rank correlation between two-run and three-run averages | 0.78 |
| Mean final accuracy: original run / five-run average | 0.661 / 0.662 |
| Success rate: original run / best objective of five runs | 51.5% / 73.5% |

Selecting the largest SGM objective among five runs recovered every success that occurred in those five runs on this pair (147 of 200 sets). This is a practical restart result for this setting, not a guarantee on other pairs. Repeating the solver reduces outcome noise, but does not eliminate seed-set differences: estimated between-set variance in final accuracy is 1.51 times the within-set variance.

Five-fold cross-validation used only the 17 existing Graph A features and these 200 sets. Against the five-run majority label, logistic regression AUC was 0.668 when trained on the original single-run label and 0.644 when trained on the five-run majority; random forest AUC was 0.600 and 0.605, respectively. Predicting the five-run mean accuracy remained weak (ridge R² about 0.12; random forest R² at most 0.05). Thus cleaner labels did not produce a clear predictive gain in this small audit. These comparisons are on one graph pair and should not be interpreted as transfer performance.

The strongest individual rank correlations with five-run mean accuracy were mean three-hop seed count among nonseeds (0.383), fraction within two hops of at least three seeds (0.335), mean pairwise seed distance (-0.317), and seed-neighborhood signature entropy (0.308). They are descriptive associations, not independent feature effects.

The 1,000 SGM runs took 1,012 seconds in total, about 17 minutes. The full original 1,000-set dataset at five runs per set would take roughly 85 minutes at the same speed.

The run-level data and complete numeric summary are in `../data/solver_variance/paper_solver_variance_runs.csv` and `../data/solver_variance/paper_solver_variance_summary.json`. The saved test sets were not used in this audit.
