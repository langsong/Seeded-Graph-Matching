"""Record how a collection was run: library versions, threads, git state."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
VERSIONED_PACKAGES = ("numpy", "scipy", "networkx", "graspologic", "scikit-learn")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def git_state() -> dict[str, object]:
    """Current commit and whether tracked files differ from it (None if unknown)."""

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "--no-optional-locks", *args], cwd=REPOSITORY_ROOT,
            capture_output=True, text=True, timeout=10,
        )

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        commit = status = None
    if commit is None or commit.returncode != 0:
        return {"git_commit": None, "git_tracked_changes": None}
    return {
        "git_commit": commit.stdout.strip(),
        "git_tracked_changes": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def collect_environment(blas_threads: int | None) -> dict[str, object]:
    """Library versions and thread settings, which can change SGM outcomes."""

    versions = {}
    for package in VERSIONED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        from threadpoolctl import threadpool_info

        threadpools = [
            {key: pool.get(key) for key in ("internal_api", "num_threads", "version")}
            for pool in threadpool_info()
        ]
    except ImportError:
        threadpools = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "package_versions": versions,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "blas_threads_limit": blas_threads,
        "threadpools": threadpools,
        **git_state(),
    }


def thread_limit(blas_threads: int | None):
    """Context manager limiting BLAS/OpenMP threads (no-op if None or 0)."""

    if not blas_threads:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        warnings.warn(
            "threadpoolctl is not installed; only the thread environment "
            "variables limit threads.",
            stacklevel=2,
        )
        return contextlib.nullcontext()
    return threadpool_limits(limits=blas_threads)


def write_json(path: Path, content: dict) -> None:
    """Write JSON atomically, so a crash never leaves a half-written file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
