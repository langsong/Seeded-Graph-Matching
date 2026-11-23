"""Convert the real graphs in Data/ into NumPy archives, check them, and summarise them.

Run from the repository root:

    python -m real_data_experiments.prepare_data              # convert, check, summarise (seconds)
    python -m real_data_experiments.prepare_data --faq-check  # also rerun two SGM-paper numbers (about a minute)

Writes real_data_experiments/data/: one .npz per dataset, manifest.json (raw-file
hashes, array shapes, notes) and summary.csv (one row per standard pair).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import datasets as D

def _enron_directed_edges_with_loops() -> tuple[int, int]:
    """Directed edge counts of weeks 130 and 131, self-loops included, as iGraphMatch counts them."""
    pair = D.get_pair("enron", directed=True, loops=True)
    return int((pair.A != 0).sum()), int((pair.B != 0).sum())


def _enron_components() -> tuple[int, int]:
    summary = D.get_pair("enron").summary()
    return summary["A_largest_component"], summary["B_largest_component"]


# Published facts the converted data should reproduce: (label, value, expected).
EXPECTED = [
    ("worm: 279 neurons", lambda: D.get_pair("worm").n, 279),
    ("worm: 2,194 directed chemical edges (Chen et al. 2016)",
     lambda: D.get_pair("worm", directed=True).summary()["A_edges"], 2194),
    ("worm: 514 electrical edges (Chen et al. 2016)", lambda: D.get_pair("worm").summary()["B_edges"], 514),
    ("worm: 26 neurons without gap junctions (Chen et al. 2016)",
     lambda: D.get_pair("worm").summary()["B_isolated"], 26),
    ("worm: electrical largest component 248 (SGM paper)",
     lambda: D.get_pair("worm").summary()["B_largest_component"], 248),
    ("worm: 253 neurons without isolates (Patsolic et al.)",
     lambda: D.get_pair("worm", vertices="nonisolated").n, 253),
    ("wikipedia: 1,382 articles", lambda: D.get_pair("wikipedia").n, 1382),
    ("wikipedia: English graph connected",
     lambda: D.get_pair("wikipedia").summary()["A_largest_component"], 1382),
    ("wikipedia: French largest component 1,323 (SGM paper)",
     lambda: D.get_pair("wikipedia").summary()["B_largest_component"], 1323),
    ("enron: weeks 130 / 131 have 488 / 482 directed edges incl. self-loops (iGraphMatch)",
     _enron_directed_edges_with_loops, (488, 482)),
    ("enron: weeks 130 / 131 largest components 146 / 151 (iGraphMatch)", _enron_components, (146, 151)),
]


def run_checks() -> bool:
    ok = True
    for label, compute, expected in EXPECTED:
        value = compute()
        value = tuple(int(v) for v in value) if isinstance(value, tuple) else int(value)
        passed = value == expected
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label}: got {value}")
    return ok


def seeded_faq_accuracy(pair: D.RealPair, n_seeds: int, replicate: int) -> float:
    """One seeded FAQ run (scipy's implementation) on uniformly random seeds.

    Returns the share of non-seeds matched correctly. Used only as a sanity check
    against published numbers; the experiments use experiment_common.sgm.
    """
    from scipy.optimize import quadratic_assignment

    rng = np.random.default_rng([replicate, n_seeds, pair.n])
    A, B, truth = pair.for_matching(shuffle_seed=int(rng.integers(2 ** 31)))
    seeds = rng.choice(pair.n, size=n_seeds, replace=False)
    result = quadratic_assignment(
        A, B, method="faq",
        options={"maximize": True, "partial_match": np.column_stack([seeds, truth[seeds]])},
    )
    nonseeds = np.setdiff1d(np.arange(pair.n), seeds)
    return float(np.mean(np.asarray(result.col_ind)[nonseeds] == truth[nonseeds]))


def run_faq_check(replicates: int) -> None:
    cases = [
        ("worm, all 279, 200 random seeds", D.get_pair("worm"), 200,
         "SGM paper: about 16% of the remaining 79 neurons"),
        ("wikipedia, all 1,382, 50 random seeds", D.get_pair("wikipedia"), 50,
         "SGM paper: above 50%"),
    ]
    for label, pair, n_seeds, published in cases:
        started = time.perf_counter()
        scores = [seeded_faq_accuracy(pair, n_seeds, r) for r in range(replicates)]
        print(f"  {label}: mean {np.mean(scores):.3f} (range {min(scores):.3f}-{max(scores):.3f}, "
              f"{replicates} runs, {time.perf_counter() - started:.0f} s). {published}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(D.RAW_DIR), help="folder with the downloaded files")
    parser.add_argument("--data-dir", default=str(D.DATA_DIR), help="where the archives are written")
    parser.add_argument("--faq-check", action="store_true",
                        help="rerun seeded FAQ on the worm and Wikipedia and compare with the SGM paper")
    parser.add_argument("--replicates", type=int, default=5, help="runs per --faq-check case")
    args = parser.parse_args()

    manifest = D.prepare(args.raw_dir, args.data_dir)
    print(f"Wrote {len(manifest['datasets'])} archives and manifest.json to {args.data_dir}")
    for name, info in manifest["datasets"].items():
        for note in info["notes"]:
            print(f"  [{name}] {note}")

    print("\nChecks against published numbers:")
    if args.data_dir != str(D.DATA_DIR):
        print("  (checks read the default data folder; skipped for a custom --data-dir)")
        ok = True
    else:
        ok = run_checks()

    import pandas as pd

    table = D.summary_table(args.data_dir)
    table.to_csv(f"{args.data_dir}/summary.csv", index=False)
    columns = ["pair", "n", "A_edges", "B_edges", "A_mean_degree", "B_mean_degree", "A_degree_cv",
               "B_degree_cv", "A_isolated", "B_isolated", "common_edges", "edge_correlation",
               "share_of_A_edges_in_B", "degree_spearman"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nStandard pairs (also in summary.csv):")
        print(table[columns].round(3).to_string(index=False))

    if args.faq_check:
        print("\nSeeded FAQ sanity check (scipy's FAQ, uniformly random seeds):")
        run_faq_check(args.replicates)
    if not ok:
        raise SystemExit("Some checks failed; see above.")


if __name__ == "__main__":
    main()
