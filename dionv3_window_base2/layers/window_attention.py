from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.layers.attention import LinearKMaskedBias, rope_apply


def window_partition(x: Tensor, window_size: int) -> Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int, B: int) -> Tensor:
    C = windows.shape[-1]
    windows = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(B, H, W, C)


class WindowSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        shift_size: int = 0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def get_attention_mask(self, H: int, W: int, device, dtype, batch_size: int) -> Tensor:
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        for h in range(0, H, self.window_size):
            for w in range(0, W, self.window_size):
                img_mask[:, h : h + self.window_size, w : w + self.window_size, :] = cnt
                cnt += 1
        shifted_mask = torch.roll(img_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        mask_windows = window_partition(shifted_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0)
        attn_mask = attn_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return attn_mask.reshape(-1, self.window_size * self.window_size, self.window_size * self.window_size).to(dtype)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, hw: Tuple[int, int], rope: Tensor | Tuple[Tensor, Tensor] | None = None) -> Tensor:
        B, N, _ = x.shape
        H, W = hw
        patch_tokens = x.view(B, H, W, -1)

        if self.shift_size > 0:
            shifted_tokens = torch.roll(patch_tokens, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_tokens = patch_tokens

        x_windows = window_partition(shifted_tokens, self.window_size)
        attn_mask = None
        if self.shift_size > 0:
            attn_mask = self.get_attention_mask(H, W, x.device, x.dtype, B)

        attn_windows = self.compute_attention(x_windows, attn_mask=attn_mask, rope=rope)
        merged = window_reverse(attn_windows, self.window_size, H, W, B)

        if self.shift_size > 0:
            merged = torch.roll(merged, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        merged = merged.view(B, H * W, -1)
        return merged

    def compute_attention(self, qkv: Tensor, attn_mask=None, rope=None) -> Tensor:
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = self.qkv(qkv).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


__all__ = ["WindowSelfAttention", "window_partition", "window_reverse"]
