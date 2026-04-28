from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import requests
import torch

from vjepa2_1_mlx.constants import CHECKPOINTS_DIR, CONVERTED_DIR, MODEL_SPECS, VJEPA_BASE_URL
from vjepa2_1_mlx.utils.fs import ensure_artifact_dirs


def checkpoint_path_for_model(model_name: str) -> Path:
    return CHECKPOINTS_DIR / MODEL_SPECS[model_name]["checkpoint_name"]


def converted_path_for_model(model_name: str) -> Path:
    checkpoint_name = MODEL_SPECS[model_name]["checkpoint_name"].replace(".pt", ".safetensors")
    return CONVERTED_DIR / checkpoint_name


def download_file(url: str, destination: Path) -> Path:
    ensure_artifact_dirs()
    destination.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    expected_size = int(head.headers.get("content-length", "0"))
    if destination.exists():
        if expected_size == 0 or destination.stat().st_size == expected_size:
            return destination
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with partial.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    partial.rename(destination)
    return destination


def download_checkpoint(model_name: str) -> Path:
    spec = MODEL_SPECS[model_name]
    return download_file(f"{VJEPA_BASE_URL}/{spec['checkpoint_name']}", checkpoint_path_for_model(model_name))


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def extract_encoder_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "ema_encoder" in checkpoint:
        state_dict = checkpoint["ema_encoder"]
    elif "encoder" in checkpoint:
        state_dict = checkpoint["encoder"]
    elif "target_encoder" in checkpoint:
        state_dict = checkpoint["target_encoder"]
    else:
        raise KeyError(f"unsupported checkpoint structure: {checkpoint.keys()}")
    return clean_state_dict(state_dict)


def torch_tensor_to_mlx_array(key: str, value: torch.Tensor) -> mx.array:
    array = value.detach().cpu().numpy()
    if array.ndim == 5 and key.endswith("weight"):
        array = array.transpose(0, 2, 3, 4, 1)
    elif array.ndim == 4 and key.endswith("weight"):
        array = array.transpose(0, 2, 3, 1)
    return mx.array(array)


def convert_checkpoint_to_mlx(checkpoint_path: str | Path, destination: str | Path) -> Path:
    state_dict = extract_encoder_state(checkpoint_path)
    mlx_state = {}
    for key, value in state_dict.items():
        mlx_state[key] = torch_tensor_to_mlx_array(key, value)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mx.eval(list(mlx_state.values()))
    mx.save_safetensors(str(destination), mlx_state)
    return destination


def ensure_converted_checkpoint(model_name: str) -> Path:
    converted_path = converted_path_for_model(model_name)
    if converted_path.exists():
        return converted_path
    checkpoint_path = download_checkpoint(model_name)
    return convert_checkpoint_to_mlx(checkpoint_path, converted_path)
