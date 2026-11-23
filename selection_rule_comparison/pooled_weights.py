"""The frozen pooled weights (per SD of random sets) for every algorithm and graph model.

Moved unchanged from best_of_n.py (now archive/best_of_n.py) on 23 Nov 2026. The weights were fitted in
fixed_pair_seed_set_classification/multi_pair_models.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO_ROOT / "fixed_pair_seed_set_classification" / "data" / "multi_pair_2026-10-30" / "pooled_weights.json"
CORE = ("degree", "harmonic", "pair_mass", "uniqueness")


def load_weights(path: Path | None = None) -> dict:
    """Pooled weights per SD, keyed by (algorithm, model)."""

    saved = json.loads(Path(path or WEIGHTS_PATH).read_text(encoding="utf-8"))
    weights = {}
    for key, entry in saved.items():
        algorithm, model = key.split("/")
        weights[(algorithm, model)] = {f: float(entry["weights_per_sd"][f]) for f in CORE}
    return weights
