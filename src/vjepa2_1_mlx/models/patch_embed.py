from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 4:
            raise ValueError(f"expected 4D BCHW image tensor, got shape={x.shape}")
        x = x.transpose(0, 2, 3, 1)
        x = self.proj(x)
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, x.shape[-1])


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 5:
            raise ValueError(f"expected 5D BCTHW video tensor, got shape={x.shape}")
        x = x.transpose(0, 2, 3, 4, 1)
        x = self.proj(x)
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, x.shape[-1])
