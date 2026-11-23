# Historical REU experiments

This folder contains the original experiment drivers, first-direction notebooks, and saved `results/`. The shared graph generators remain in `utils/Graphs.py`; the percolation algorithm is in `experiment_common/percolation.py`.

Run the historical sweep from the repository root with `python -m legacy_reu.experiment_sweep`. New plots and JSON files are written to `legacy_reu/results/`.

The first-direction notebooks were moved here for archival organization. To rerun their cells later, start Jupyter with the repository root as the working directory so their existing imports resolve.
