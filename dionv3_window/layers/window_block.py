from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn

from dinov3.layers.ffn_layers import Mlp
from dinov3.layers.layer_scale import LayerScale

from .window_attention import WindowSelfAttention


torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class WindowSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowSelfAttention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

    def _forward_single(self, x: Tensor, hw: Tuple[int, int], rope=None) -> Tensor:
        x_attn = x + self.ls1(self.attn(self.norm1(x), hw=hw, rope=rope))
        x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
        return x_ffn

    def forward(
        self,
        x_or_x_list,
        rope_or_rope_list=None,
        hw_or_hw_list: Optional[List[Tuple[int, int]] | Tuple[int, int]] = None,
    ):
        if isinstance(x_or_x_list, Tensor):
            assert isinstance(hw_or_hw_list, tuple)
            return self._forward_single(x_or_x_list, hw_or_hw_list, rope_or_rope_list)
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in x_or_x_list]
            assert isinstance(hw_or_hw_list, list)
            return [self._forward_single(x, hw, rope) for x, hw, rope in zip(x_or_x_list, hw_or_hw_list, rope_or_rope_list)]
        else:
            raise AssertionError


__all__ = ["WindowSelfAttentionBlock"]
