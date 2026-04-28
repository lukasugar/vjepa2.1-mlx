from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from vjepa2_1_mlx.compare import compare_features
from vjepa2_1_mlx.models.vision_transformer import VisionTransformer
from vjepa2_1_mlx.utils.checkpoints import torch_tensor_to_mlx_array


def _load_upstream_module():
    upstream_root = Path(__file__).resolve().parents[1] / "references" / "vjepa2"
    upstream_root_str = str(upstream_root)
    if upstream_root_str not in sys.path:
        sys.path.insert(0, upstream_root_str)
    from app.vjepa_2_1.models.vision_transformer import VisionTransformer as TorchVisionTransformer  # pyright: ignore[reportMissingImports]

    return TorchVisionTransformer


def _build_torch_model():
    torch.manual_seed(0)
    TorchVisionTransformer = _load_upstream_module()
    model = TorchVisionTransformer(
        img_size=(32, 32),
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=96,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_sdpa=False,
        use_silu=False,
        wide_silu=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
        n_output_distillation=1,
    )
    model.eval()
    return model


def _build_mlx_model():
    torch.manual_seed(0)
    model = VisionTransformer(
        img_size=(32, 32),
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=96,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_silu=False,
        wide_silu=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
        n_output_distillation=1,
    )
    weights = [(key, torch_tensor_to_mlx_array(key, value)) for key, value in _build_torch_model().state_dict().items()]
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def _compare_on_input(input_tensor: np.ndarray) -> dict[str, float]:
    torch_model = _build_torch_model()
    mlx_model = _build_mlx_model()
    with torch.inference_mode():
        torch_output = torch_model(torch.from_numpy(input_tensor)).numpy()
    mlx_output = np.asarray(mlx_model(mx.array(input_tensor)))
    return compare_features(torch_output, mlx_output)


def test_small_image_mode_parity():
    input_tensor = np.random.default_rng(0).normal(size=(1, 3, 1, 32, 32)).astype(np.float32)
    metrics = _compare_on_input(input_tensor)
    assert metrics["cosine_similarity"] > 0.999999
    assert metrics["max_abs_error"] < 1e-4


def test_small_video_mode_parity():
    input_tensor = np.random.default_rng(1).normal(size=(1, 3, 4, 32, 32)).astype(np.float32)
    metrics = _compare_on_input(input_tensor)
    assert metrics["cosine_similarity"] > 0.999999
    assert metrics["max_abs_error"] < 1e-4
