from __future__ import annotations

import numpy as np
import mlx.core as mx

from vjepa2_1_mlx.models.vision_transformer import build_vjepa2_1_model
from vjepa2_1_mlx.utils.checkpoints import ensure_converted_checkpoint


class MLXBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = build_vjepa2_1_model(model_name)
        converted_path = ensure_converted_checkpoint(model_name)
        self.model.load_weights(str(converted_path), strict=True)
        mx.eval(self.model.parameters())

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        tensor = mx.array(input_tensor)
        output = self.model(tensor)
        mx.eval(output)
        return np.asarray(output)
