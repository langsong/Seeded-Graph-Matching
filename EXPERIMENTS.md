# Research experiment layout

The historical REU experiment scripts, notebooks, and saved plots are in `legacy_reu/`. The graph generators in `utils/Graphs.py` remain at the repository root; the percolation implementation is shared through `experiment_common/percolation.py`.

| Folder | Research question |
|---|---|
| `experiment_common/` | Shared graph presets and pair generation, feature extraction, SGM evaluation, and percolation matching. |
| `cross_pair_seed_set_quality/` | Can seed-set features predict SGM performance across independently generated graph pairs? |
| `fixed_pair_seed_set_classification/` | On one fixed graph pair, can we distinguish good and bad seed sets without graph-pair difficulty as a confounder? |
| `one_seed_marginal_problem/` | Given an existing seed set, which Graph A vertex should be queried next? |

Each folder contains its code, notebook, data, metadata, and a README with module-based running commands.
