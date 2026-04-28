from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from vjepa2_1_mlx.constants import MODEL_SPECS
from vjepa2_1_mlx.models.modules import Block, apply_masks, trunc_normal
from vjepa2_1_mlx.models.patch_embed import PatchEmbed, PatchEmbed3D


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_path_rate: float = 0.0,
        use_silu: bool = False,
        wide_silu: bool = True,
        use_rope: bool = True,
        handle_nonsquare_inputs: bool = True,
        img_temporal_dim_size: int | None = None,
        n_registers: int = 0,
        has_cls_first: bool = False,
        interpolate_rope: bool = False,
        modality_embedding: bool = True,
        n_output_distillation: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.img_height, self.img_width = img_size
        self.is_video = num_frames > 1
        self.use_rope = use_rope

        dpr = [float(x) for x in mx.linspace(0.0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_img = None
        if self.img_temporal_dim_size is not None:
            self.patch_embed_img = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=1,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )

        self.blocks = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[i],
                wide_silu=wide_silu,
                use_silu=use_silu,
                grid_size=img_size[0] // patch_size,
                use_rope=use_rope,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
                interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
            for i in range(depth)
        ]

        if depth == 12:
            self.hierarchical_layers = [2, 5, 8, 11]
            self.out_layers_distillation = [2, 5, 8, 11] if n_output_distillation == 4 else [11]
        elif depth == 24:
            self.hierarchical_layers = [5, 11, 17, 23]
            self.out_layers_distillation = [5, 11, 17, 23] if n_output_distillation == 4 else [23]
        else:
            raise ValueError(f"unsupported depth for 2.1 encoder parity: {depth}")

        self.norms_block = [nn.LayerNorm(embed_dim, eps=1e-6) for _ in self.hierarchical_layers]
        self.return_hierarchical = False
        self.modality_embedding = modality_embedding
        if modality_embedding:
            self.img_mod_embed = trunc_normal((1, 1, embed_dim), std=1e-6)
            self.video_mod_embed = trunc_normal((1, 1, embed_dim), std=1e-6)

    def check_temporal_dim(self, shape: tuple[int, ...]) -> bool:
        return self.img_temporal_dim_size is not None and shape[2] == self.img_temporal_dim_size

    def _broadcast_parameter(self, parameter: mx.array, batch_size: int) -> mx.array:
        return mx.broadcast_to(parameter, (batch_size, parameter.shape[1], parameter.shape[2]))

    def __call__(self, x: mx.array, masks: list[mx.array] | None = None, training: bool = False) -> mx.array:
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, height, width = x.shape
            temporal = 1
        elif x.ndim == 5:
            _, _, temporal_raw, height, width = x.shape
            temporal = temporal_raw if self.check_temporal_dim(x.shape) else temporal_raw // self.tubelet_size
        else:
            raise ValueError(f"expected BCHW or BCTHW input, got shape={x.shape}")

        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        if not self.handle_nonsquare_inputs:
            temporal = h_patches = w_patches = None

        if self.check_temporal_dim(x.shape):
            if self.patch_embed_img is None:
                raise ValueError("patch_embed_img missing for image temporal path")
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x = x + self._broadcast_parameter(self.img_mod_embed, x.shape[0])
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x = x + self._broadcast_parameter(self.video_mod_embed, x.shape[0])

        if masks is not None:
            x = apply_masks(x, masks)
            masks = [mx.concatenate(masks, axis=0)] if len(masks) > 1 else masks

        hier = []
        mask_arg = masks[0] if masks else None
        for index, block in enumerate(self.blocks):
            x, _ = block(
                x,
                mask=mask_arg,
                T=temporal,
                H_patches=h_patches,
                W_patches=w_patches,
                mode=mode,
            )
            if index in self.out_layers_distillation:
                out_index = self.hierarchical_layers.index(index)
                hier.append(self.norms_block[out_index](x))

        if training or self.return_hierarchical:
            return mx.concatenate(hier, axis=2)
        return self.norms_block[-1](x)


def build_vjepa2_1_model(model_name: str) -> VisionTransformer:
    spec = MODEL_SPECS[model_name]
    return VisionTransformer(
        img_size=(spec["img_size"], spec["img_size"]),
        patch_size=spec["patch_size"],
        num_frames=spec["num_frames"],
        tubelet_size=spec["tubelet_size"],
        embed_dim=spec["embed_dim"],
        depth=spec["depth"],
        num_heads=spec["num_heads"],
        mlp_ratio=spec["mlp_ratio"],
        qkv_bias=True,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
        n_output_distillation=1,
    )
