"""A-only seed-bank sampling; no matching outcomes are used here."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def unique_sets(sets):
    return list({tuple(sorted(map(int, s))): np.sort(s) for s in sets}.values())


def swap_paths(starts, targets, rng, steps=24):
    """Several independent random-to-endpoint paths, sampled along their lengths."""
    result = []
    for start in starts:
        for target in targets:
            gone = rng.permutation(np.setdiff1d(start, target))
            added = rng.permutation(np.setdiff1d(target, start))
            keep = set(np.linspace(1, len(gone), min(steps, len(gone)), dtype=int))
            current = set(map(int, start))
            for j, (v, w) in enumerate(zip(gone, added), 1):
                current.remove(int(v))
                current.add(int(w))
                if j in keep:
                    result.append(np.asarray(sorted(current), dtype=np.int64))
    return unique_sets(result)


def bin_numbers(values, bins):
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    constant = high - low <= 1e-12 * max(1.0, abs(low), abs(high))
    edges = np.linspace(low, high, bins + 1)
    numbers = (np.zeros(len(values), dtype=int) if constant else
               np.minimum(np.searchsorted(edges, values, side="right") - 1, bins - 1))
    return numbers, edges, constant


def select_bank(arrays, values, score_columns, rule_indices, count, random_count, rng, bins=8):
    """Keep controls, endpoints, approximate degree contrasts and balanced score intervals.

    The first random_count arrays must be distinct uniform draws. Intervals use
    the entire constructed bank, not quantiles of a uniform random pool. A set
    can fill intervals for several scores. Identical memberships occur once.
    """
    arrays = np.asarray(arrays, dtype=np.int64)
    count = min(count, len(arrays))
    chosen = list(range(min(random_count, count)))
    labels = [(i, "random", "uniform", "") for i in chosen]

    def add(i, family, detail, contrast=""):
        i = int(i)
        if i not in chosen:
            if len(chosen) >= count:
                return
            chosen.append(i)
        labels.append((i, family, detail, contrast))

    for name, i in rule_indices.items():
        add(i, "rule", name)
    # Preserve extrema for every score used to stratify the collection.
    for col in score_columns:
        add(values[col].idxmax(), "endpoint", col)

    # Narrow cut-degree bands, with low/high values of another feature.
    degree = values["degree"].to_numpy()
    width = max(float(np.ptp(degree)) * 0.025, 1e-12)
    contrast_limit = min(count, len(chosen) + (30 if count >= 140 else 15))
    for band, centre in enumerate(np.linspace(degree.min(), degree.max(), 6)):
        members = np.flatnonzero(np.abs(degree - centre) <= width)
        if len(members) < 2:
            continue
        for col in ("pair_mass", "coverage", "uniqueness"):
            part = values.iloc[members][col]
            lo, hi = int(part.idxmin()), int(part.idxmax())
            if lo == hi or values.at[lo, col] == values.at[hi, col]:
                continue
            new = sum(i not in chosen for i in (lo, hi))
            if len(chosen) + new > contrast_limit:
                continue
            contrast = f"degree_band_{band}:{col}"
            add(lo, "contrast", f"low_{col}", contrast)
            add(hi, "contrast", f"high_{col}", contrast)
            # Retain memberships even if a contrast endpoint was already chosen.
            for i, side in ((lo, "low"), (hi, "high")):
                labels.append((i, "contrast_member", f"{side}_{col}", contrast))

    intervals = {}
    for col in score_columns:
        numbers, edges, constant = bin_numbers(values[col], bins)
        if not constant:
            for b in range(bins):
                members = np.flatnonzero(numbers == b)
                if len(members):
                    intervals[col, b] = members
    tie_order = {key: j for j, key in enumerate(rng.permutation(len(intervals)))}
    interval_keys = list(intervals)
    # Membership indicators make overlap computations cheap for a shortlist.
    membership = np.zeros((len(arrays), int(arrays.max()) + 1), dtype=np.float32)
    membership[np.arange(len(arrays))[:, None], arrays] = 1
    while len(chosen) < count:
        available = np.ones(len(arrays), dtype=bool)
        available[chosen] = False
        options = []
        for j, key in enumerate(interval_keys):
            members = intervals[key]
            left = members[available[members]]
            if len(left):
                options.append((len(members) - len(left), tie_order[j], j, left))
        if not options:
            remaining = np.flatnonzero(available)
            if not len(remaining):
                break
            add(rng.choice(remaining), "range", "remaining_bank")
            continue
        _, _, j, left = min(options, key=lambda x: x[:2])
        col, b = interval_keys[j]
        shortlist = rng.choice(left, min(32, len(left)), replace=False)
        if chosen:
            overlap = membership[shortlist] @ membership[chosen].T
            distance = 1 - overlap.max(axis=1) / arrays.shape[1]
            i = shortlist[int(np.argmax(distance))]
        else:
            i = shortlist[0]
        add(i, "range", f"{col}:bin_{b}")

    selected = values.iloc[chosen].copy().reset_index(drop=True)
    selected.insert(0, "candidate_id", np.arange(len(selected)))
    selected["seed_vertices"] = [json.dumps(arrays[i].tolist()) for i in chosen]
    mapping = {old: new for new, old in enumerate(chosen)}
    members = pd.DataFrame([
        {"candidate_id": mapping[i], "family": family, "detail": detail, "contrast_id": contrast}
        for i, family, detail, contrast in labels if i in mapping
    ]).drop_duplicates()
    primary = members.drop_duplicates("candidate_id").set_index("candidate_id")
    selected["family"] = selected.candidate_id.map(primary.family)
    selected["detail"] = selected.candidate_id.map(primary.detail)
    coverage = []
    random_indices = np.arange(min(random_count, len(arrays)))
    for col in score_columns:
        numbers, edges, constant = bin_numbers(values[col], bins)
        for b in range(bins):
            coverage.append({"score": col, "bin": b, "lower": edges[b], "upper": edges[b + 1],
                             "constant_score": constant, "available": int(np.sum(numbers == b)),
                             "selected": int(np.sum(numbers[chosen] == b)),
                             "random_controls": int(np.sum(numbers[random_indices] == b))})
    return selected, members, pd.DataFrame(coverage)
