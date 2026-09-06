"""Run provenance: hashes, package versions, git commit, manifests.

Model selection is only trustworthy if a reported number can be tied back to an
exact config, dataset and code state. Every artifact-producing stage writes a
:class:`RunManifest` next to its output.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "gymnasium",
    "polars",
    "pandas",
    "pydantic",
    "typer",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable hash of any JSON-serialisable object (sorted keys)."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def sha256_array(*arrays: np.ndarray) -> str:
    """Hash of one or more arrays including dtype and shape.

    Shape/dtype are folded in so that a reshape or dtype change cannot collide
    with the original data.
    """
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "absent"
    return out


def git_commit(repo_root: str | Path | None = None) -> str:
    """Best-effort git SHA; ``"unknown"`` when not a repo or git is absent."""
    cwd = str(repo_root) if repo_root else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
        if sha.returncode != 0:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
        suffix = "-dirty" if dirty.stdout.strip() else ""
        return sha.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class RunManifest:
    """Everything needed to re-derive an artifact."""

    stage: str
    seed: int
    config_hash: str
    config: dict[str, Any]
    data_hash: str = ""
    split_boundaries: list[dict[str, Any]] = field(default_factory=list)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=package_versions)
    git_commit: str = field(default_factory=git_commit)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p

    @staticmethod
    def read(path: str | Path) -> RunManifest:
        data = json.loads(Path(path).read_text())
        return RunManifest(**data)
