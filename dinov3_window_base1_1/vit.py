from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from dinov3.utils import named_apply
from dionv3_window.layers.window_block import WindowSelfAttentionBlock


ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformerWindowBaseline1_1(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = 2.0,
        pos_embed_rope_dtype: str = "fp32",
        embed_dim: int = 4096,
        depth: int = 40,
        num_heads: int = 32,
        ffn_ratio: float = 3.0,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = 1e-5,
        norm_layer: str = "layernormbf16",
        ffn_layer: str = "swiglu64",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 4,
        mask_k_bias: bool = True,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = True,
        window_size: int = 7,
        global_block_indices: Optional[Sequence[int]] = None,
        global_block_interval: int = 3,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            print(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth

        if global_block_indices is not None:
            global_block_set = set(global_block_indices)
        else:
            global_block_set = None

        blocks_list = []
        for i in range(depth):
            is_global = False
            if global_block_set is not None:
                is_global = i in global_block_set
            elif global_block_interval > 0:
                is_global = (i + 1) % global_block_interval == 0

            if is_global:
                blocks_list.append(
                    SelfAttentionBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        ffn_ratio=ffn_ratio_sequence[i],
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        drop_path=drop_path_rate,
                        norm_layer=norm_layer_cls,
                        act_layer=nn.GELU,
                        ffn_layer=ffn_layer_cls,
                        init_values=layerscale_init,
                        mask_k_bias=mask_k_bias,
                        device=device,
                    )
                )
            else:
                shift_size = 0 if i % 2 == 0 else window_size // 2
                blocks_list.append(
                    _PatchOnlyWindowBlock(
                        global_token_count=self.n_storage_tokens + 1,
                        dim=embed_dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=shift_size,
                        ffn_ratio=ffn_ratio_sequence[i],
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        drop=drop_path_rate,
                        norm_layer=norm_layer_cls,
                        act_layer=nn.GELU,
                        ffn_layer=ffn_layer_cls,
                        init_values=layerscale_init,
                        mask_k_bias=mask_k_bias,
                        device=device,
                    )
                )

        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        hw_list = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            hw_list.append(hw_tuple)
        for blk in self.blocks:
            if isinstance(blk, SelfAttentionBlock):
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in hw_list]
                x = blk(x, rope_sincos)
            else:
                rope_sincos = [None for _ in x]
                x = blk(x, rope_sincos, hw_list)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if isinstance(blk, SelfAttentionBlock):
                rope_sincos = self.rope_embed(H=H, W=W)
                x = blk(x, rope_sincos)
            else:
                rope_sincos = None
                x = blk(x, rope_sincos, (H, W))
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        else:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, x: torch.Tensor | List[torch.Tensor], masks: Optional[torch.Tensor] = None) -> List[Dict[str, torch.Tensor]]:
        return self.forward_features(x, masks)


class _PatchOnlyWindowBlock(nn.Module):
    """Window attention applied only to patch tokens.

    CLS/storage tokens bypass the attention and FFN and are forwarded unchanged, so
    they are effectively used only in global blocks.
    """

    def __init__(
        self,
        *,
        global_token_count: int,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        ffn_ratio: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        drop: float,
        norm_layer,
        act_layer,
        ffn_layer,
        init_values,
        mask_k_bias,
        device,
    ) -> None:
        super().__init__()
        self.global_token_count = global_token_count
        self.patch_block = WindowSelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            mask_k_bias=mask_k_bias,
            device=device,
        )

    def _forward_single(self, x: Tensor, rope, hw: Tuple[int, int]) -> Tensor:
        g_tokens = x[:, : self.global_token_count]
        p_tokens = x[:, self.global_token_count :]
        p_tokens = self.patch_block(p_tokens, rope, hw)
        return torch.cat([g_tokens, p_tokens], dim=1)

    def forward(
        self,
        x_or_x_list,
        rope_or_rope_list=None,
        hw_or_hw_list: Optional[List[Tuple[int, int]] | Tuple[int, int]] = None,
    ):
        if isinstance(x_or_x_list, Tensor):
            assert isinstance(hw_or_hw_list, tuple)
            return self._forward_single(x_or_x_list, rope_or_rope_list, hw_or_hw_list)
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in x_or_x_list]
            assert isinstance(hw_or_hw_list, list)
            return [
                self._forward_single(x, rope, hw)
                for x, hw, rope in zip(x_or_x_list, hw_or_hw_list, rope_or_rope_list)
            ]
        else:
            raise AssertionError


__all__ = ["DinoVisionTransformerWindowBaseline1_1"]
