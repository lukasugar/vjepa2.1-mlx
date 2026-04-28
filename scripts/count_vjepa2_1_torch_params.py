#!/usr/bin/env python3
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vjepa2_1_mlx.constants import MODEL_SPECS, UPSTREAM_VJEPA2_DIR


def _ensure_upstream_imports() -> None:
    upstream_root = REPO_ROOT / UPSTREAM_VJEPA2_DIR
    expected_model_file = upstream_root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
    if not expected_model_file.exists():
        raise FileNotFoundError(
            "Expected the upstream V-JEPA 2 repo at "
            f"{upstream_root}, but {expected_model_file} is missing."
        )

    upstream_root_str = str(upstream_root)
    if upstream_root_str not in sys.path:
        sys.path.insert(0, upstream_root_str)


def _build_model(model_name: str):
    _ensure_upstream_imports()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        from app.vjepa_2_1.models import vision_transformer as vit_encoder  # pyright: ignore[reportMissingImports]

    spec = MODEL_SPECS[model_name]
    constructor = getattr(vit_encoder, spec["arch_name"])
    return constructor(
        img_size=(spec["img_size"], spec["img_size"]),
        patch_size=spec["patch_size"],
        num_frames=spec["num_frames"],
        tubelet_size=spec["tubelet_size"],
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )


def main() -> None:
    for model_name in ("vjepa2_1_vit_base_384", "vjepa2_1_vit_large_384"):
        model = _build_model(model_name)
        params = sum(param.numel() for param in model.parameters())
        short_name = "b" if "base" in model_name else "l"
        print(f"{short_name} ({model_name}): {params:,} parameters")


if __name__ == "__main__":
    main()
