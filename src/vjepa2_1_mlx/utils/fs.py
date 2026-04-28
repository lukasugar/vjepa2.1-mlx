from __future__ import annotations

from pathlib import Path

from vjepa2_1_mlx.constants import (
    ARTIFACTS_DIR,
    BENCHMARKS_DIR,
    CHECKPOINTS_DIR,
    CONVERTED_DIR,
    FEATURES_DIR,
    SAMPLES_DIR,
)


def ensure_artifact_dirs() -> None:
    for path in (
        ARTIFACTS_DIR,
        CHECKPOINTS_DIR,
        SAMPLES_DIR,
        CONVERTED_DIR,
        FEATURES_DIR,
        BENCHMARKS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved
