from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from vjepa2_1_mlx.constants import MODEL_SPECS, UPSTREAM_VJEPA2_COMMIT, UPSTREAM_VJEPA2_DIR
from vjepa2_1_mlx.utils.checkpoints import download_checkpoint, extract_encoder_state


def _ensure_upstream_imports() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    upstream_root = repo_root / UPSTREAM_VJEPA2_DIR
    expected_model_file = upstream_root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
    if not expected_model_file.exists():
        script_path = repo_root / "scripts" / "clone_reference_vjepa2.sh"
        raise FileNotFoundError(
            "PyTorch backend requires the upstream V-JEPA 2 reference repo at "
            f"{upstream_root}. Expected file missing: {expected_model_file}. "
            f"Bootstrap it with `{script_path}`. "
            f"The pinned upstream commit for this repo is {UPSTREAM_VJEPA2_COMMIT}."
        )
    upstream_root_str = str(upstream_root)
    if upstream_root_str not in sys.path:
        sys.path.insert(0, upstream_root_str)


def _build_encoder(model_name: str):
    _ensure_upstream_imports()
    from app.vjepa_2_1.models import vision_transformer as vit_encoder  # pyright: ignore[reportMissingImports]

    spec = MODEL_SPECS[model_name]
    constructor = getattr(vit_encoder, spec["arch_name"])
    encoder = constructor(
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
    return encoder


class PyTorchBackend:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = _build_encoder(model_name)
        checkpoint_path = download_checkpoint(model_name)
        state_dict = extract_encoder_state(checkpoint_path)
        message = self.model.load_state_dict(state_dict, strict=True)
        if message.missing_keys or message.unexpected_keys:
            raise RuntimeError(f"strict checkpoint load failed: {message}")
        self.model.to(self.device)
        self.model.eval()

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(input_tensor).to(self.device, dtype=torch.float32)
            output = self.model(tensor)
            return output.detach().cpu().numpy()
