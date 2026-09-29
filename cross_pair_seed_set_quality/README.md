# Cross-pair seed-set quality

This experiment samples seed sets across many independently generated graph pairs. It asks whether Graph A seed-set features and correspondence-free graph context predict H0 or final matching accuracy.

- `data_collection.py`: SGM data collection and feature extraction.
- `percolation_data_collection.py`: parallel data collection for percolation matching.
- `regressor.ipynb`: exploration, regression, and classification.
- `data/`: collected CSV files and metadata.

Run commands from the repository root:

```bash
python -m cross_pair_seed_set_quality.data_collection --graph-model all
python -m cross_pair_seed_set_quality.percolation_data_collection --graph-model all
```

For the new preferential-attachment-plus-ER PAPER model, run:

```bash
python -m cross_pair_seed_set_quality.data_collection --graph-model paper_pa
```

This writes `data/seed_quality_paper_pa.csv`; set `GRAPH_MODEL = 'paper_pa'` in `regressor.ipynb` to run the same regression and classification analysis. Its starting preset is 600 vertices, attachment weights `alpha=0, beta=1`, ER probability `p=0.005`, ER-edge correlation `rho=0.62`, and 20 seeds. A small pilot on one fixed pair found both successes and failures; the success mix across other pairs remains to be measured. The older `paper` option and saved `seed_quality_paper.csv` remain the Pareto-weight Chung–Lu model, despite their historical name. The percolation collector still covers only its previously calibrated models.

Existing datasets are protected; add `--overwrite` only when intentionally regenerating them.
