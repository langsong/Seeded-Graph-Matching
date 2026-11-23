"""Layout checks after the 23 Nov 2026 reorganization.

Every module imports, archived experiments still use the package's data folder, and every notebook's imports resolve.
Run from the repository root: python -m unittest selection_rule_comparison.tests.test_layout
"""

from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ("experiment_common", "selection_rule_comparison", "real_data_experiments")
EXTRA = ("fixed_pair_seed_set_classification.multi_pair_collection", "fixed_pair_seed_set_classification.sbm_collection")
NOTEBOOK_FOLDERS = ("selection_rule_comparison/notebooks", "selection_rule_comparison/archive",
                    "real_data_experiments/notebooks", "fixed_pair_seed_set_classification")
OUR_IMPORTS = ("from selection_rule_comparison", "from real_data_experiments", "from fixed_pair_seed_set_classification",
               "from experiment_common")


def module_names():
    for package in PACKAGES:
        for path in sorted((REPO / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            parts = list(path.relative_to(REPO).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            yield ".".join(parts)
    yield from EXTRA


class Layout(unittest.TestCase):
    def test_every_module_imports(self):
        failures = []
        for name in module_names():
            try:
                importlib.import_module(name)
            except Exception as error:  # noqa: BLE001 - report every failure
                failures.append(f"{name}: {type(error).__name__}: {error}")
        self.assertEqual(failures, [])

    def test_archived_experiments_use_the_package_data_folder(self):
        package = REPO / "selection_rule_comparison"
        names = ("best_of_n", "beyond_top_degree", "borderline_check", "hub_test", "pilot_new_rules",
                 "real_graph_check", "regime_sweep", "sbm_rules", "score_transfer", "weekend_queue")
        for name in names:
            module = importlib.import_module(f"selection_rule_comparison.archive.{name}")
            self.assertEqual(Path(module.EXPERIMENT_DIR).resolve(), package, name)
        audit = importlib.import_module("selection_rule_comparison.archive.temperature_audit")
        self.assertEqual(audit.DEFAULT_OUTPUT.resolve(), package / "data" / "temperature_audit")
        self.assertEqual(audit.ROOT.resolve(), REPO)

    def test_notebook_imports_resolve(self):
        failures = []
        for folder in NOTEBOOK_FOLDERS:
            for path in sorted((REPO / folder).glob("*.ipynb")):
                for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    for line in "".join(cell["source"]).splitlines():
                        line = line.strip()
                        if not line.startswith(OUR_IMPORTS) or " import " not in line or line.endswith(("(", "\\")):
                            continue
                        try:
                            exec(line, {})
                        except Exception as error:  # noqa: BLE001
                            failures.append(f"{path.relative_to(REPO)}: {line}: {type(error).__name__}: {error}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
