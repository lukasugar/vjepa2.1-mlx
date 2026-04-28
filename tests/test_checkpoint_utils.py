from __future__ import annotations

import numpy as np
import torch

from vjepa2_1_mlx.utils.checkpoints import torch_tensor_to_mlx_array


def test_conv_weight_transposes_match_mlx_layout():
    conv2d = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    conv3d = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)

    converted2d = np.asarray(torch_tensor_to_mlx_array("patch_embed.proj.weight", conv2d))
    converted3d = np.asarray(torch_tensor_to_mlx_array("patch_embed_img.proj.weight", conv3d))

    assert converted2d.shape == (2, 4, 5, 3)
    assert converted3d.shape == (2, 4, 5, 6, 3)
    np.testing.assert_array_equal(converted2d, conv2d.numpy().transpose(0, 2, 3, 1))
    np.testing.assert_array_equal(converted3d, conv3d.numpy().transpose(0, 2, 3, 4, 1))
