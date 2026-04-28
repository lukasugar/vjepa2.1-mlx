from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def trunc_normal(shape: tuple[int, ...], std: float = 0.02) -> mx.array:
    values = mx.random.normal(shape=shape) * std
    return mx.clip(values, -2 * std, 2 * std)


def apply_masks(x: mx.array, masks: list[mx.array] | None) -> mx.array:
    if masks is None:
        return x
    kept = []
    for mask in masks:
        kept.append(mx.take(x, mask, axis=1))
    return mx.concatenate(kept, axis=0)


def repeat_interleave_last_dim(x: mx.array, repeats: int) -> mx.array:
    x = mx.expand_dims(x, axis=-1)
    x = mx.repeat(x, repeats, axis=-1)
    return x.reshape(*x.shape[:-2], -1)


def rotate_queries_or_keys(
    x: mx.array,
    pos: mx.array,
    n_registers: int,
    has_cls_first: bool,
) -> mx.array:
    _, _, num_tokens, dim = x.shape
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even")

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = num_tokens - n_registers

    parts = []
    if n_cls:
        parts.append(x[..., :n_cls, :])

    x_ctx = x[..., start_ctx:end_ctx, :]
    omega = mx.arange(dim // 2, dtype=x.dtype)
    omega = omega / (dim / 2.0)
    omega = 1.0 / (10000**omega)
    freq = mx.expand_dims(pos, axis=-1) * omega

    emb_sin = repeat_interleave_last_dim(mx.sin(freq), 2)
    emb_cos = repeat_interleave_last_dim(mx.cos(freq), 2)

    y = x_ctx.reshape(*x_ctx.shape[:-1], -1, 2)
    y1 = y[..., 0]
    y2 = y[..., 1]
    y = mx.stack((-y2, y1), axis=-1).reshape(x_ctx.shape)
    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)
    parts.append(out_ctx)

    if n_registers:
        parts.append(x[..., end_ctx:, :])

    return mx.concatenate(parts, axis=-2)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def __call__(self, x: mx.array) -> mx.array:
        if self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = mx.random.bernoulli(keep_prob, shape).astype(x.dtype)
        return x * mask / keep_prob


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = drop

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        wide_silu: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        swiglu_hidden_features = hidden_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            swiglu_hidden_features = ((swiglu_hidden_features + 7) // 8) * 8
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc3(nn.silu(self.fc1(x)) * self.fc2(x))


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        grid_size: int = 14,
        is_causal: bool = False,
        n_registers: int = 0,
        has_cls_first: bool = False,
        interpolate_rope: bool = False,
        patch_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.d_dim = int(2 * ((self.head_dim // 3) // 2))
        self.h_dim = int(2 * ((self.head_dim // 3) // 2))
        self.w_dim = int(2 * ((self.head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        if patch_size == 14:
            self.pretrained_grid_size = 18
        elif patch_size == 16:
            self.pretrained_grid_size = 16
        else:
            self.pretrained_grid_size = grid_size

    def _get_frame_pos(self, ids: mx.array, h_patches: int | None, w_patches: int | None) -> mx.array:
        tokens_per_frame = int((h_patches or self.grid_size) * (w_patches or self.grid_size))
        return ids // tokens_per_frame

    def _get_height_pos(self, ids: mx.array, h_patches: int | None, w_patches: int | None) -> mx.array:
        tokens_per_frame = int((h_patches or self.grid_size) * (w_patches or self.grid_size))
        tokens_per_row = int(w_patches or self.grid_size)
        frame_ids = self._get_frame_pos(ids, h_patches, w_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(
        self, ids: mx.array, h_patches: int | None, w_patches: int | None
    ) -> tuple[mx.array, mx.array, mx.array]:
        tokens_per_frame = int((h_patches or self.grid_size) * (w_patches or self.grid_size))
        tokens_per_row = int(w_patches or self.grid_size)
        frame_ids = self._get_frame_pos(ids, h_patches, w_patches)
        height_ids = self._get_height_pos(ids, h_patches, w_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids.astype(mx.float32), height_ids.astype(mx.float32), width_ids.astype(mx.float32)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        T: int | None = None,
        H_patches: int | None = None,
        W_patches: int | None = None,
        return_attn: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, -1)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mx.expand_dims(mask, axis=1)
            mask = mx.repeat(mask, self.num_heads, axis=1)
        else:
            if T is None or H_patches is None or W_patches is None:
                ctx = int((num_tokens - self.n_registers))
                mask = mx.arange(ctx, dtype=mx.int32)
            else:
                mask = mx.arange(int(T * H_patches * W_patches), dtype=mx.int32)

        d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        if self.interpolate_rope:
            local_h = H_patches or self.grid_size
            local_w = W_patches or self.grid_size
            if local_h > 1:
                h_mask = h_mask * (self.pretrained_grid_size - 1) / (local_h - 1)
            if local_w > 1:
                w_mask = w_mask * (self.pretrained_grid_size - 1) / (local_w - 1)

        split = 0
        qd = rotate_queries_or_keys(q[..., split : split + self.d_dim], d_mask, self.n_registers, self.has_cls_first)
        kd = rotate_queries_or_keys(k[..., split : split + self.d_dim], d_mask, self.n_registers, self.has_cls_first)
        split += self.d_dim
        qh = rotate_queries_or_keys(q[..., split : split + self.h_dim], h_mask, self.n_registers, self.has_cls_first)
        kh = rotate_queries_or_keys(k[..., split : split + self.h_dim], h_mask, self.n_registers, self.has_cls_first)
        split += self.h_dim
        qw = rotate_queries_or_keys(q[..., split : split + self.w_dim], w_mask, self.n_registers, self.has_cls_first)
        kw = rotate_queries_or_keys(k[..., split : split + self.w_dim], w_mask, self.n_registers, self.has_cls_first)
        split += self.w_dim

        if split < self.head_dim:
            q = mx.concatenate([qd, qh, qw, q[..., split:]], axis=-1)
            k = mx.concatenate([kd, kh, kw, k[..., split:]], axis=-1)
        else:
            q = mx.concatenate([qd, qh, qw], axis=-1)
            k = mx.concatenate([kd, kh, kw], axis=-1)

        attn = None
        if return_attn:
            attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            if self.is_causal:
                causal = mx.triu(mx.full((num_tokens, num_tokens), -1e9, dtype=attn.dtype), k=1)
                attn = attn + causal
            attn = mx.softmax(attn, axis=-1)
            x = attn @ v
        else:
            attn_mask = "causal" if self.is_causal else None
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        return x, attn if return_attn else None


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        is_causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.is_causal = is_causal

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_mask = "causal" if self.is_causal else None
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, num_tokens, channels)
        return self.proj(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_path: float = 0.0,
        wide_silu: bool = True,
        use_silu: bool = False,
        grid_size: int = 16,
        use_rope: bool = False,
        is_causal: bool = False,
        n_registers: int = 0,
        has_cls_first: bool = False,
        interpolate_rope: bool = False,
        patch_size: int = 16,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPEAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                grid_size=grid_size,
                is_causal=is_causal,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
                interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
        else:
            self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, is_causal=is_causal)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_silu:
            self.mlp = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, wide_silu=wide_silu)
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def _apply_drop_path(self, x: mx.array) -> mx.array:
        if self.drop_path is None:
            return x
        return self.drop_path(x)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        T: int | None = None,
        H_patches: int | None = None,
        W_patches: int | None = None,
        return_attn: bool = False,
        mode: str = "video",
    ) -> tuple[mx.array, mx.array | None]:
        del mode
        if self.use_rope:
            y, attn = self.attn(
                self.norm1(x),
                mask=mask,
                T=T,
                H_patches=H_patches,
                W_patches=W_patches,
                return_attn=return_attn,
            )
        else:
            y = self.attn(self.norm1(x))
            attn = None
        x = x + self._apply_drop_path(y)
        x = x + self._apply_drop_path(self.mlp(self.norm2(x)))
        return x, attn
