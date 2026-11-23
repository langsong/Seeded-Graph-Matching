"""Real graph pairs for seeded graph matching experiments.

The raw files sit in the repository's Data/ folder (downloaded from
https://www.cis.jhu.edu/~parky/SGM/). `prepare_data.py` converts them once into
NumPy archives in real_data_experiments/data/, and everything here reads those.

    from real_data_experiments.datasets import get_pair

    pair = get_pair("worm")                      # chemical vs electrical, 279 neurons
    A, B, truth = pair.for_matching(shuffle_seed=0)

`for_matching` returns the same triple as experiment_common.graphs.generate_graph_pair:
dense float adjacency matrices and `truth`, where truth[v] is the vertex of B that
matches vertex v of A. So `run_sgm_once(A, B, truth, seeds, ...)` and
`run_percolation_once(A, B, truth, seeds, ...)` take a real pair unchanged.

A `RealPair` stores the two graphs aligned (vertex i of A is vertex i of B).
Always match the shuffled copy from `for_matching`: with B left aligned, ties in
the assignment step can land on the right answer by accident.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
RAW_DIR = REPO_ROOT / "Data"
DATA_DIR = PACKAGE_DIR / "data"

DATASETS = {
    "worm": "C. elegans hermaphrodite, chemical vs electrical (gap junction) synapses, 279 neurons",
    "worm_male": "C. elegans male (posterior), chemical vs electrical, the 167 neurons named in both",
    "wikipedia": "Wikipedia 'Algebraic Geometry' articles, English vs French hyperlinks, 1,382 articles",
    "enron": "Enron email, one week vs another, 184 addresses (weeks are 1-based, as in the SGM paper)",
}
ARCHIVES = {
    "worm": "worm_hermaphrodite.npz",
    "worm_male": "worm_male.npz",
    "wikipedia": "wikipedia_en_fr.npz",
    "enron": "enron_weekly.npz",
}
RAW_FILES = {
    "worm": ("elegansGraph.mat", "herm_Graph.Rd"),
    "worm_male": ("elegansGraph_male.mat",),
    "wikipedia": ("GE.Rdata", "GF.Rdata"),
    "enron": ("enron.mat",),
}
# 'vcols' in herm_Graph.Rd, as documented on the SGM data page.
WORM_CLASSES = {1: "motor", 2: "interneuron", 3: "sensory"}
VERTEX_RULES = ("all", "nonisolated", "common_lcc")


# --------------------------------------------------------------------------
# Raw files -> archives
# --------------------------------------------------------------------------

def _cell_strings(cells) -> list[str]:
    """MATLAB cell array of strings -> list of str."""
    return [str(np.ravel(cell)[0]) if np.size(cell) else "" for cell in np.ravel(cells)]


def _counts(matrix) -> np.ndarray:
    """Dense integer copy of a synapse-count matrix (checks the counts are whole)."""
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    dense = dense.astype(float)
    if not np.array_equal(dense, np.round(dense)):
        raise ValueError("Expected whole-number synapse counts.")
    return dense.astype(np.int32)


def read_raw(raw_dir: str | Path = RAW_DIR) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[str]]]:
    """Read the downloaded files into plain arrays, checking them against each other.

    Returns the arrays for each dataset and notes on the choices made.
    """

    from scipy.io import loadmat
    from .rdata_reader import read_rdata

    raw_dir = Path(raw_dir)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    notes: dict[str, list[str]] = {}

    # C. elegans hermaphrodite. Edges and synapse counts come from elegansGraph.mat;
    # neuron classes come from herm_Graph.Rd, after checking both files list the
    # same neurons in the same order with the same edges.
    mat = loadmat(raw_dir / "elegansGraph.mat")
    rds = read_rdata(raw_dir / "herm_Graph.Rd")
    names = _cell_strings(mat["Neuron_ordered"])
    if names != [str(name) for name in rds["neurons"]]:
        raise ValueError("elegansGraph.mat and herm_Graph.Rd list different neurons.")
    chemical, electrical = _counts(mat["Achem"]), _counts(mat["Agap"])
    r_chemical, r_electrical = np.asarray(rds["Ac"]), np.asarray(rds["Ag"])
    if not np.array_equal(chemical != 0, r_chemical != 0) or not np.array_equal(electrical, r_electrical):
        raise ValueError("elegansGraph.mat and herm_Graph.Rd disagree on the edges.")
    differing = int(((chemical != r_chemical) & (chemical != 0)).sum())
    codes = np.asarray(rds["vcols"], dtype=int)
    arrays["worm"] = {
        "chemical": chemical,          # directed, chemical[i, j] = synapses from i to j
        "electrical": electrical,      # symmetric gap-junction counts (3 self-loops)
        "names": np.array(names),
        "neuron_class": np.array([WORM_CLASSES[code] for code in codes]),
        "neuron_type_code": np.array([str(code) for code in rds["types"]]),
        "Q": np.asarray(rds["Q"], dtype=float),  # undocumented 3-column table in herm_Graph.Rd
    }
    notes["worm"] = [
        "Both files list the same 279 neurons in the same order and the same edges.",
        f"Synapse counts are taken from elegansGraph.mat (whole numbers); herm_Graph.Rd's chemical "
        f"weights differ on {differing} of {int((chemical != 0).sum())} edges (some are halved), so "
        "weighted analyses should state which file they use.",
        "neuron_class decodes herm_Graph.Rd 'vcols' as 1 = motor, 2 = interneuron, 3 = sensory "
        "(SGM data page). 'types' and 'Q' are kept unlabelled.",
    ]

    # C. elegans male: two graphs on slightly different neuron lists.
    male = loadmat(raw_dir / "elegansGraph_male.mat")
    arrays["worm_male"] = {
        "chemical": _counts(male["Achem"]),
        "electrical": _counts(male["Agap"]),
        "chemical_names": np.array(_cell_strings(male["Achem_names"])),
        "electrical_names": np.array(_cell_strings(male["Agap_names"])),
    }
    notes["worm_male"] = [
        "Chemical has 170 neurons and electrical 167; get_pair keeps those named in both, after "
        "removing '*' from names ('AN1b*' and '*AN1b' are taken to be the same neuron).",
    ]

    # Wikipedia: GE and GF hold shortest-path distances (0 on the diagonal), so a
    # hyperlink is a distance of 1. Row i of both files is the same article.
    english = np.asarray(read_rdata(raw_dir / "GE.Rdata")["GE"])
    french = np.asarray(read_rdata(raw_dir / "GF.Rdata")["GF"])
    for label, distances in (("GE", english), ("GF", french)):
        if not np.array_equal(distances, distances.T) or np.diag(distances).any():
            raise ValueError(f"{label} is not a symmetric distance matrix with a zero diagonal.")
    arrays["wikipedia"] = {
        "english": (english == 1).astype(np.uint8),
        "french": (french == 1).astype(np.uint8),
    }
    notes["wikipedia"] = [
        "GE.Rdata and GF.Rdata store shortest-path distances, not adjacency; edges are distance 1.",
        "GF uses 6 for some long or unreachable distances; that does not affect the edges.",
    ]

    # Enron: 184 x 184 x 187 directed weekly email graphs, entry [i, j, t] as stored.
    weekly = np.asarray(loadmat(raw_dir / "enron.mat")["AAA"])
    arrays["enron"] = {"weekly": (weekly != 0).astype(np.uint8)}
    notes["enron"] = [
        "Directed, with self-loops (self-sent email); weeks are numbered from 1 as in MATLAB and "
        "the SGM paper. Weeks 130 and 131 are the pair iGraphMatch ships as `Enron`.",
    ]
    return arrays, notes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(raw_dir: str | Path = RAW_DIR, data_dir: str | Path = DATA_DIR) -> dict:
    """Convert the raw files into data_dir/*.npz and write data_dir/manifest.json."""

    import scipy

    raw_dir, data_dir = Path(raw_dir), Path(data_dir)
    missing = [f for files in RAW_FILES.values() for f in files if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {raw_dir}: {', '.join(missing)}")
    arrays, notes = read_raw(raw_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "raw_dir": os.path.relpath(raw_dir.absolute(), REPO_ROOT),
        "datasets": {},
    }
    for name, contents in arrays.items():
        np.savez_compressed(data_dir / ARCHIVES[name], **contents)
        manifest["datasets"][name] = {
            "description": DATASETS[name],
            "archive": ARCHIVES[name],
            "raw_files": {f: _sha256(raw_dir / f) for f in RAW_FILES[name]},
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       for key, value in contents.items()},
            "notes": notes[name],
        }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_arrays(name: str, data_dir: str | Path = DATA_DIR) -> dict[str, np.ndarray]:
    """The arrays stored for one dataset."""

    if name not in ARCHIVES:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}.")
    path = Path(data_dir) / ARCHIVES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m real_data_experiments.prepare_data` from the repository root."
        )
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


# --------------------------------------------------------------------------
# Graph pairs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RealPair:
    """Two graphs on the same vertices, aligned: vertex i of A is vertex i of B."""

    name: str
    A: np.ndarray  # the graph seeds are chosen from (float64)
    B: np.ndarray  # the other graph, aligned with A (float64)
    labels: np.ndarray  # vertex names (neuron names; row indices for Wikipedia and Enron)
    settings: dict = field(default_factory=dict)
    vertex_class: np.ndarray | None = None  # worm only: motor / interneuron / sensory

    @property
    def n(self) -> int:
        return self.A.shape[0]

    def for_matching(self, shuffle_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A, shuffled B, truth) in the experiment_common convention.

        B is relabelled by a random permutation; truth[v] is the vertex of the
        shuffled B that matches vertex v of A.
        """
        shuffle = np.random.default_rng(shuffle_seed).permutation(self.n)
        return self.A.copy(), self.B[np.ix_(shuffle, shuffle)].copy(), np.argsort(shuffle)

    def summary(self) -> dict:
        return {"pair": self.name, **describe(self.A, self.B)}


def _normalised_name(name: str) -> str:
    return str(name).strip().replace("*", "")


def _simplify(matrix: np.ndarray, weighted: bool, directed: bool, loops: bool) -> np.ndarray:
    """Apply the edge conventions to one adjacency matrix."""
    M = np.asarray(matrix, dtype=float).copy()
    diagonal = np.diag(M).copy()
    if not directed and not np.array_equal(M, M.T):
        M = M + M.T if weighted else np.maximum(M, M.T)
    np.fill_diagonal(M, diagonal if loops else 0.0)
    if not weighted:
        M = (M != 0).astype(float)
    return M


def _undirected(M: np.ndarray) -> np.ndarray:
    S = (M != 0) | (M.T != 0)
    np.fill_diagonal(S, False)
    return S


def degrees(M: np.ndarray) -> np.ndarray:
    """Number of distinct neighbours, ignoring direction, weights and self-loops."""
    return _undirected(M).sum(axis=1)


def largest_component(M: np.ndarray) -> np.ndarray:
    """Boolean mask of the vertices in the largest (weakly) connected component."""
    from scipy.sparse.csgraph import connected_components

    _, labels = connected_components(_undirected(M).astype(np.int8), directed=False)
    return labels == np.argmax(np.bincount(labels))


def _vertex_subset(A: np.ndarray, B: np.ndarray, rule: str) -> np.ndarray:
    """Vertices kept by a rule, re-applied until nothing more is removed."""
    if rule not in VERTEX_RULES:
        raise ValueError(f"vertices must be one of {VERTEX_RULES}, not {rule!r}.")
    keep = np.arange(A.shape[0])
    if rule == "all":
        return keep
    while True:
        a, b = A[np.ix_(keep, keep)], B[np.ix_(keep, keep)]
        if rule == "nonisolated":
            mask = (degrees(a) > 0) & (degrees(b) > 0)
        else:
            mask = largest_component(a) & largest_component(b)
        if mask.all():
            return keep
        keep = keep[mask]


def get_pair(
    name: str,
    *,
    graph_a: str | None = None,
    weeks: tuple[int, int] = (130, 131),
    weighted: bool = False,
    directed: bool = False,
    loops: bool = False,
    vertices: str = "all",
    data_dir: str | Path = DATA_DIR,
) -> RealPair:
    """Build an aligned graph pair.

    name      "worm", "worm_male", "wikipedia" or "enron".
    graph_a   which graph is A (the graph seeds are chosen from): "chemical" (default) or
              "electrical" for the worms, "english" (default) or "french" for Wikipedia.
              For Enron, A is the first of `weeks`.
    weeks     Enron only: the two weeks to match, numbered from 1 (the SGM paper used 130-132).
    weighted  keep synapse counts (worms); otherwise every edge has weight 1.
    directed  keep edge directions (worm chemical synapses, Enron); otherwise an edge in
              either direction becomes one undirected edge (counts are added if weighted).
    loops     keep self-loops.
    vertices  "all" keeps every vertex. "nonisolated" drops vertices with no edges in
              either graph (the 253-neuron worm of Patsolic et al.), and "common_lcc"
              keeps vertices in the largest connected component of both graphs. Both
              rules are re-applied until no further vertex is removed.

    The defaults (unweighted, undirected, no loops, all vertices) are the SGM paper's setup.
    """

    arrays = load_arrays(name, data_dir)
    vertex_class = None
    if name == "worm":
        graphs = {"chemical": arrays["chemical"], "electrical": arrays["electrical"]}
        labels, vertex_class, default_a = arrays["names"], arrays["neuron_class"], "chemical"
    elif name == "worm_male":
        chemical_names = [_normalised_name(x) for x in arrays["chemical_names"]]
        electrical_index = {_normalised_name(x): i for i, x in enumerate(arrays["electrical_names"])}
        rows = [i for i, x in enumerate(chemical_names) if x in electrical_index]
        chem_rows = np.array(rows)
        elec_rows = np.array([electrical_index[chemical_names[i]] for i in rows])
        graphs = {
            "chemical": arrays["chemical"][np.ix_(chem_rows, chem_rows)],
            "electrical": arrays["electrical"][np.ix_(elec_rows, elec_rows)],
        }
        labels, default_a = np.array([chemical_names[i] for i in rows]), "chemical"
    elif name == "wikipedia":
        graphs = {"english": arrays["english"], "french": arrays["french"]}
        labels, default_a = np.arange(arrays["english"].shape[0]).astype(str), "english"
    elif name == "enron":
        weekly = arrays["weekly"]
        first, second = (int(t) for t in weeks)
        for t in (first, second):
            if not 1 <= t <= weekly.shape[2]:
                raise ValueError(f"Enron weeks run from 1 to {weekly.shape[2]}; got {t}.")
        if first == second:
            raise ValueError("Choose two different Enron weeks.")
        graphs = {f"week {first}": weekly[:, :, first - 1], f"week {second}": weekly[:, :, second - 1]}
        labels, default_a = np.arange(weekly.shape[0]).astype(str), f"week {first}"
    else:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}.")

    graph_a = graph_a or default_a
    if graph_a not in graphs:
        raise ValueError(f"graph_a must be one of {list(graphs)} for {name!r}.")
    graph_b = next(g for g in graphs if g != graph_a)
    A = _simplify(graphs[graph_a], weighted, directed, loops)
    B = _simplify(graphs[graph_b], weighted, directed, loops)
    keep = _vertex_subset(A, B, vertices)
    dropped = np.setdiff1d(np.arange(len(labels)), keep)
    settings = {
        "dataset": name, "graph_a": graph_a, "graph_b": graph_b, "weighted": weighted,
        "directed": directed, "loops": loops, "vertices": vertices,
        "n_before": int(len(labels)), "n": int(len(keep)), "dropped": labels[dropped].tolist(),
    }
    if name == "enron":
        settings["weeks"] = [first, second]
    return RealPair(
        name=name,
        A=A[np.ix_(keep, keep)],
        B=B[np.ix_(keep, keep)],
        labels=labels[keep],
        settings=settings,
        vertex_class=None if vertex_class is None else vertex_class[keep],
    )


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def describe(A: np.ndarray, B: np.ndarray) -> dict:
    """Size, degree spread and agreement of two aligned graphs.

    `edge_correlation` is the Pearson correlation between the two graphs' edge
    indicators over all vertex pairs. For a correlated Erdos-Renyi pair it equals the
    generator's rho, so it places a real pair on the regime map's rho axis (roughly:
    real graphs are heterogeneous). `share_of_A_edges_in_B` is the kept-edge fraction.
    """

    from scipy.stats import spearmanr

    n = A.shape[0]
    directed = not (np.array_equal(A, A.T) and np.array_equal(B, B.T))
    pairs = ~np.eye(n, dtype=bool) if directed else np.triu(np.ones((n, n), dtype=bool), 1)
    edges_a, edges_b = A[pairs] != 0, B[pairs] != 0
    out: dict = {"n": n, "directed": directed}
    for label, M, edges in (("A", A, edges_a), ("B", B, edges_b)):
        deg = degrees(M)
        out[f"{label}_edges"] = int(edges.sum())
        out[f"{label}_density"] = float(edges.mean())
        out[f"{label}_mean_degree"] = float(deg.mean())
        out[f"{label}_max_degree"] = int(deg.max())
        out[f"{label}_degree_cv"] = float(deg.std() / deg.mean()) if deg.mean() > 0 else float("nan")
        out[f"{label}_isolated"] = int((deg == 0).sum())
        out[f"{label}_largest_component"] = int(largest_component(M).sum())
    common = int((edges_a & edges_b).sum())
    out["common_edges"] = common
    out["edge_correlation"] = float(np.corrcoef(edges_a, edges_b)[0, 1])
    out["share_of_A_edges_in_B"] = common / max(out["A_edges"], 1)
    out["degree_spearman"] = float(spearmanr(degrees(A), degrees(B)).correlation)
    return out


STANDARD_PAIRS = (
    ("worm, all 279", dict(name="worm")),
    ("worm, non-isolated", dict(name="worm", vertices="nonisolated")),
    ("worm, common LCC", dict(name="worm", vertices="common_lcc")),
    ("worm, directed", dict(name="worm", directed=True)),
    ("worm male", dict(name="worm_male")),
    ("wikipedia, all", dict(name="wikipedia")),
    ("wikipedia, common LCC", dict(name="wikipedia", vertices="common_lcc")),
    ("enron 130 vs 131", dict(name="enron", weeks=(130, 131))),
    ("enron 131 vs 132", dict(name="enron", weeks=(131, 132))),
    ("enron 130 vs 132", dict(name="enron", weeks=(130, 132))),
    ("enron 130 vs 131, common LCC", dict(name="enron", weeks=(130, 131), vertices="common_lcc")),
)


def summary_table(data_dir: str | Path = DATA_DIR):
    """One row per standard pair (see STANDARD_PAIRS)."""

    import pandas as pd

    rows = []
    for label, kwargs in STANDARD_PAIRS:
        pair = get_pair(**kwargs, data_dir=data_dir)
        rows.append({"pair": label, "graph_a": pair.settings["graph_a"],
                     "graph_b": pair.settings["graph_b"], **describe(pair.A, pair.B)})
    return pd.DataFrame(rows)
