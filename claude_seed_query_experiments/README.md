# Seed-query experiments (Claude, Sep 2026)

Self-contained experiments on **which correspondences to pay for**: given Graph A, a budget of
k queries (each query reveals the true Graph-B partner of one Graph-A vertex) and possibly some
already verified matches, which vertices should be queried so that SGM is most accurate?

Nothing outside this folder is modified. The code imports nothing from the rest of the repo;
the fixed pairs and saved outcomes it reuses are read from `../fixed_pair_seed_set_classification/data`
and `../one_seed_marginal_problem/data` (paths set at the top of each script).

**Read `REPORT.md` first.**

## Important implementation note

graspologic could not be installed in the environment these experiments ran in, so
`sgmlib.py` re-implements the same FAQ/SGM solver (barycenter start, Frank-Wolfe with exact
line search, final linear assignment; graspologic-style random tie breaking). It was checked
against `scipy.optimize.quadratic_assignment(method="faq")` (identical up to floating-point
tie breaking) and against the saved graspologic runs (see REPORT.md, "Validation").
Graph generators follow `utils/Graphs.py` / `cross_pair_seed_set_quality.preset_config`.

## Files

| File | What it does |
|---|---|
| `sgmlib.py` | SGM solver, graph generators, Graph-A statistics, selection rules |
| `validate_scipy.py`, `validate2.py`, `validate3.py` | solver equivalence checks against scipy's FAQ |
| `e0_protocol.py`, `analyze_e0.py` | iteration cap, restarts, per-iteration trajectories, per-vertex witness curves (saved fixed pairs) |
| `e1_static_rules.py`, `analyze_e1.py` | static rules at equal budget on fresh pairs, 5 regimes x 5 budgets |
| `e1b_focused.py`, `analyze_e1b.py` | focused head-to-head at the most discriminating budget, 12 new pairs per regime |
| `e2_adaptive.py` | batch-sequential (active) querying from verified matches vs static rules |
| `e2b_adaptive_replication.py` | replication of the best adaptive rule on new pairs, with a static control |
| `e3_one_seed.py`, `analyze_e3.py` | the repo's one-seed marginal problems re-run with solver restarts |
| `e4_enron.py` | real Enron email pairs (weekly pair, 4-week windows, edge-subsampled parent) |
| `analyze_e2_e4.py`, `make_figures.py` | analysis tables and report figures |
| `results/` | raw run-level CSVs (one row per SGM run: `acc` = non-seed accuracy after up to 100 iterations, `acc30` = at iteration 30, `h0` = first-direction accuracy, `score` = objective, `seeds` = Graph-A seed list, `correct` = packed per-vertex correctness) |
| `results/pairs.zip` | the generated graph pairs; unzip into `results/` to reuse them exactly (the scripts regenerate them from fixed seeds if missing) |
| `figures/` | PNG figures used in the report |

## Re-running

```bash
cd claude_seed_query_experiments
pip install numpy scipy scikit-learn networkx pandas matplotlib
python validate_scipy.py            # solver equivalence
python e0_protocol.py && python analyze_e0.py
python e1_static_rules.py && python analyze_e1.py iter100
python e1b_focused.py && python analyze_e1b.py
python e2_adaptive.py && python e2b_adaptive_replication.py
python e3_one_seed.py && python analyze_e3.py
python e4_enron.py && python analyze_e2_e4.py
python make_figures.py
```

Scripts use 2 worker processes; total runtime on 2 CPU cores was about 1.5 hours. Run them from inside this folder (`python e1_static_rules.py`), so that `import sgmlib` resolves.
Set `OMP_NUM_THREADS=1` to avoid BLAS oversubscription.
