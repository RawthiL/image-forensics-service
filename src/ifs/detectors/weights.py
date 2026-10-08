"""Resolving a checkpoint to a local file.

Three sources, named explicitly in configuration. There is no silent
substitution: if the configured source cannot produce the configured file, this
raises. Nothing is ever loaded "instead".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

Source = Literal["auto", "local", "hub"]


class WeightsError(RuntimeError):
    """The configured weights could not be produced. Never falls back."""


@dataclass(frozen=True)
class ResolvedWeights:
    path: Path
    source: str  # "local" | "hub" | "path"
    revision: str | None
    """Commit sha when the file came from the hub; None for a local file."""


def resolve_path(path: str | os.PathLike[str]) -> ResolvedWeights:
    """An explicit checkpoint path. Exists or raises -- never substituted."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise WeightsError(
            f"checkpoint not found at {p}. detector.path must point at an existing file; "
            f"no download or catalog lookup is attempted for an explicit path."
        )
    log.info("weights: explicit path %s", p)
    return ResolvedWeights(path=p, source="path", revision=None)


def resolve_catalog(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    source: Source = "auto",
    weights_dir: str | os.PathLike[str] | None = None,
    allow_download: bool = True,
) -> ResolvedWeights:
    """A catalog entry, from a local directory and/or the Hugging Face hub.

    `source`:
      * ``local`` -- only `<weights_dir>/<repo_id>/<filename>`; raises if absent.
      * ``hub``   -- only the HF cache/hub; raises if unavailable.
      * ``auto``  -- local first, then hub. The *same* file either way; this is
        a lookup order, not a substitution of one model for another.
    """
    local_candidate = _local_candidate(weights_dir, repo_id, filename)

    if source == "local":
        if local_candidate is None:
            raise WeightsError(
                "weights.source is 'local' but weights.dir is not set; "
                "set weights.dir or use source 'auto'/'hub'."
            )
        if not local_candidate.is_file():
            raise WeightsError(
                f"weights.source is 'local' and {local_candidate} does not exist. "
                f"Place the checkpoint there, or change weights.source."
            )
        log.info("weights: local file %s", local_candidate)
        return ResolvedWeights(path=local_candidate, source="local", revision=None)

    if source == "auto" and local_candidate is not None and local_candidate.is_file():
        log.info("weights: local file %s", local_candidate)
        return ResolvedWeights(path=local_candidate, source="local", revision=None)

    if not allow_download:
        where = f" ({local_candidate} absent)" if local_candidate else ""
        raise WeightsError(
            f"weights.allow_download is false and {repo_id}/{filename} is not cached"
            f"{where}. Pre-populate the HF cache or weights.dir, or allow downloads."
        )

    return _from_hub(repo_id=repo_id, filename=filename, revision=revision)


def _local_candidate(
    weights_dir: str | os.PathLike[str] | None, repo_id: str, filename: str
) -> Path | None:
    """`<dir>/<org>/<repo>/<filename>`, mirroring the repo id so one directory
    can hold several models without their filenames colliding."""
    if not weights_dir:
        return None
    return Path(weights_dir).expanduser() / repo_id / filename


def _from_hub(*, repo_id: str, filename: str, revision: str) -> ResolvedWeights:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise WeightsError(
            "huggingface-hub is required to resolve a catalog model from the hub"
        ) from exc

    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    except Exception as exc:
        raise WeightsError(
            f"could not fetch {repo_id}/{filename}@{revision[:12]} from the Hugging Face hub: {exc}"
        ) from exc

    log.info("weights: hub file %s (%s@%s)", path, repo_id, revision[:12])
    return ResolvedWeights(path=Path(path), source="hub", revision=revision)
