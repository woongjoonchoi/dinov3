from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SwiGLUFFN
from dinov3.layers.attention import rope_apply
from dinov3.utils import named_apply
from dionv3_window.layers.window_attention import window_partition, window_reverse


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


class LocalGlobalBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_global_tokens: int,
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
        enable_patch_to_global: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_global_tokens = num_global_tokens
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.local_attn = WindowSelfAttentionWithGlobal(
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

        linear_q_class = nn.Linear
        linear_kv_class = nn.Linear

        self.global_q = linear_q_class(dim, dim, bias=qkv_bias, device=device)
        self.global_kv = linear_kv_class(dim, dim * 2, bias=qkv_bias, device=device)
        self.global_proj = nn.Linear(dim, dim, bias=proj_bias, device=device)

        self.enable_patch_to_global = enable_patch_to_global
        if enable_patch_to_global:
            self.patch_q = linear_q_class(dim, dim, bias=qkv_bias, device=device)
            self.patch_kv = linear_kv_class(dim, dim * 2, bias=qkv_bias, device=device)
            self.patch_proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        else:
            self.patch_q = None
            self.patch_kv = None
            self.patch_proj = None

        self.proj_drop = nn.Dropout(drop)

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

    def _global_attention(self, q_tokens: Tensor, kv_tokens: Tensor) -> Tensor:
        B, G, _ = q_tokens.shape
        q = self.global_q(q_tokens).reshape(B, G, self.num_heads, -1).transpose(1, 2)
        kv = self.global_kv(kv_tokens).reshape(B, kv_tokens.shape[1], 2, self.num_heads, -1)
        k, v = torch.unbind(kv, dim=2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, G, -1)
        out = self.global_proj(out)
        out = self.proj_drop(out)
        return out

    def _patch_global_attention(self, q_tokens: Tensor, kv_tokens: Tensor) -> Tensor:
        if not self.enable_patch_to_global:
            return torch.zeros_like(q_tokens)
        B, N, _ = q_tokens.shape
        q = self.patch_q(q_tokens).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        kv = self.patch_kv(kv_tokens).reshape(B, kv_tokens.shape[1], 2, self.num_heads, -1)
        k, v = torch.unbind(kv, dim=2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.patch_proj(out)
        out = self.proj_drop(out)
        return out

    def _forward_single(self, x: Tensor, hw: Tuple[int, int], rope=None) -> Tensor:
        B, N, _ = x.shape
        assert N >= self.num_global_tokens
        x_norm = self.norm1(x)

        global_tokens = x_norm[:, : self.num_global_tokens, :]
        patch_tokens = x_norm[:, self.num_global_tokens :, :]

        patch_local = self.local_attn(
            patch_tokens,
            hw=hw,
            rope=rope,
            global_tokens=global_tokens,
        )

        kv_tokens = torch.cat((global_tokens, patch_local), dim=1)
        global_out = self._global_attention(global_tokens, kv_tokens)

        patch_global = self._patch_global_attention(patch_local, global_out)
        patch_out = patch_local + patch_global

        attn_out = torch.cat((global_out, patch_out), dim=1)
        x_attn = x + self.ls1(attn_out)
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


class LocalGlobalHybridVisionTransformer(nn.Module):
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
        device: Any | None = None,
        enable_patch_to_global: bool = True,
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
        self.num_global_tokens = n_storage_tokens + 1

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

        blocks_list = []
        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            blocks_list.append(
                LocalGlobalBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    num_global_tokens=self.num_global_tokens,
                    window_size=window_size,
                    shift_size=shift_size,
                    ffn_ratio=ffn_ratio_sequence[i],
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=drop_path_rate,
                    attn_drop=0.0,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                    enable_patch_to_global=enable_patch_to_global,
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

    def _normalize_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Strip common wrappers (DDP/module/backbone prefixes) from checkpoint keys."""

        def strip_prefix(key: str) -> str:
            prefixes = [
                "module.",
                "backbone.",
                "model.",
            ]
            for p in prefixes:
                if key.startswith(p):
                    return key[len(p) :]
            return key

        return {strip_prefix(k): v for k, v in state_dict.items()}

    def _remap_pretrained_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Remap original DINOv3 qkv/proj weights to the Baseline3 module layout."""

        state_dict = self._normalize_state_dict(state_dict)

        def maybe_chunk_qkv(prefix: str):
            qkv_w = state_dict.get(f"{prefix}attn.qkv.weight")
            if qkv_w is None:
                return None
            qkv_b = state_dict.get(f"{prefix}attn.qkv.bias")
            q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
            if qkv_b is None:
                zeros = torch.zeros(q_w.shape[0], device=qkv_w.device, dtype=qkv_w.dtype)
                q_b, k_b, v_b = zeros, zeros, zeros
            else:
                q_b, k_b, v_b = qkv_b.chunk(3, dim=0)
            return (q_w, k_w, v_w, q_b, k_b, v_b)

        remapped = dict(state_dict)
        for i in range(len(self.blocks)):
            prefix = f"blocks.{i}."
            chunks = maybe_chunk_qkv(prefix)
            if chunks is None:
                continue
            q_w, k_w, v_w, q_b, k_b, v_b = chunks

            proj_w = state_dict.get(f"{prefix}attn.proj.weight")
            proj_b = state_dict.get(f"{prefix}attn.proj.bias")
            if proj_b is None:
                proj_b = torch.zeros_like(q_b)

            # Local branch weights
            remapped[f"{prefix}local_attn.q.weight"] = q_w.clone()
            remapped[f"{prefix}local_attn.q.bias"] = q_b.clone()
            remapped[f"{prefix}local_attn.k_local.weight"] = k_w.clone()
            remapped[f"{prefix}local_attn.k_local.bias"] = k_b.clone()
            remapped[f"{prefix}local_attn.v_local.weight"] = v_w.clone()
            remapped[f"{prefix}local_attn.v_local.bias"] = v_b.clone()
            remapped[f"{prefix}local_attn.k_global.weight"] = k_w.clone()
            remapped[f"{prefix}local_attn.k_global.bias"] = k_b.clone()
            remapped[f"{prefix}local_attn.v_global.weight"] = v_w.clone()
            remapped[f"{prefix}local_attn.v_global.bias"] = v_b.clone()

            # Global branch weights
            remapped[f"{prefix}global_q.weight"] = q_w.clone()
            remapped[f"{prefix}global_q.bias"] = q_b.clone()
            remapped[f"{prefix}global_kv.weight"] = torch.cat((k_w, v_w), dim=0)
            remapped[f"{prefix}global_kv.bias"] = torch.cat((k_b, v_b), dim=0)
            if proj_w is not None:
                remapped[f"{prefix}global_proj.weight"] = proj_w
            if proj_b is not None:
                remapped[f"{prefix}global_proj.bias"] = proj_b

            # Patch->global branch weights
            remapped[f"{prefix}patch_q.weight"] = q_w.clone()
            remapped[f"{prefix}patch_q.bias"] = q_b.clone()
            remapped[f"{prefix}patch_kv.weight"] = torch.cat((k_w, v_w), dim=0)
            remapped[f"{prefix}patch_kv.bias"] = torch.cat((k_b, v_b), dim=0)
            if proj_w is not None:
                remapped[f"{prefix}patch_proj.weight"] = proj_w
            if proj_b is not None:
                remapped[f"{prefix}patch_proj.bias"] = proj_b

            # Local projection
            if proj_w is not None:
                remapped[f"{prefix}local_attn.proj.weight"] = proj_w
            if proj_b is not None:
                remapped[f"{prefix}local_attn.proj.bias"] = proj_b

            # Remove old keys to avoid "unexpected" errors when strict=True
            remapped.pop(f"{prefix}attn.qkv.weight", None)
            remapped.pop(f"{prefix}attn.qkv.bias", None)
            remapped.pop(f"{prefix}attn.proj.weight", None)
            remapped.pop(f"{prefix}attn.proj.bias", None)

        return remapped

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        remapped = self._remap_pretrained_state_dict(state_dict)

        model_state = self.state_dict()
        filtered: Dict[str, Tensor] = {}
        for k, v in model_state.items():
            if k in remapped and isinstance(remapped[k], torch.Tensor):
                filtered[k] = remapped[k]
            else:
                filtered[k] = v
        return super().load_state_dict(filtered, strict=strict)

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
            rope_sincos = [self.rope_embed(H=H, W=W) for H, W in hw_list]
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
            rope_sincos = self.rope_embed(H=H, W=W)
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


__all__ = ["LocalGlobalHybridVisionTransformer"]
class WindowSelfAttentionWithGlobal(nn.Module):
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

        linear_q_class = nn.Linear
        linear_kv_class = nn.Linear
        self.q = linear_q_class(dim, dim, bias=qkv_bias, device=device)
        self.k_local = linear_kv_class(dim, dim, bias=qkv_bias, device=device)
        self.v_local = linear_kv_class(dim, dim, bias=qkv_bias, device=device)
        self.k_global = linear_kv_class(dim, dim, bias=qkv_bias, device=device)
        self.v_global = linear_kv_class(dim, dim, bias=qkv_bias, device=device)

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
        sin_len = sin.shape[-2]
        if sin_len > N:
            sin = sin[:, :, sin_len - N :, :]
            cos = cos[:, :, sin_len - N :, :]
            sin_len = N
        prefix = N - sin_len
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(
        self,
        patch_tokens: Tensor,
        hw: Tuple[int, int],
        rope: Tensor | Tuple[Tensor, Tensor] | None = None,
        global_tokens: Tensor | None = None,
    ) -> Tensor:
        B, N, C = patch_tokens.shape
        H, W = hw
        assert N == H * W, "patch tokens must match spatial dims"

        patch_tokens = patch_tokens.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_tokens = torch.roll(patch_tokens, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_tokens = patch_tokens

        rope_windows = None
        if rope is not None:
            sin, cos = rope
            sin = sin.view(1, H, W, -1)
            cos = cos.view(1, H, W, -1)
            if self.shift_size > 0:
                sin = torch.roll(sin, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                cos = torch.roll(cos, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            sin_windows = window_partition(sin, self.window_size).view(-1, self.window_size * self.window_size, sin.shape[-1])
            cos_windows = window_partition(cos, self.window_size).view(-1, self.window_size * self.window_size, cos.shape[-1])
            num_windows = sin_windows.shape[0]
            sin_windows = sin_windows.expand(B, num_windows, -1, -1).reshape(
                B * num_windows, self.window_size * self.window_size, sin.shape[-1]
            )
            cos_windows = cos_windows.expand(B, num_windows, -1, -1).reshape(
                B * num_windows, self.window_size * self.window_size, cos.shape[-1]
            )
            rope_windows = (sin_windows, cos_windows)

        patch_windows = window_partition(shifted_tokens, self.window_size)
        num_windows = patch_windows.shape[0] // B

        attn_mask = None
        if self.shift_size > 0:
            attn_mask_local = self.get_attention_mask(H, W, patch_tokens.device, patch_tokens.dtype, B)
            if global_tokens is not None:
                zeros_for_global = torch.zeros(
                    attn_mask_local.shape[0],
                    attn_mask_local.shape[1],
                    global_tokens.shape[1],
                    device=attn_mask_local.device,
                    dtype=attn_mask_local.dtype,
                )
                attn_mask = torch.cat([attn_mask_local, zeros_for_global], dim=-1)
            else:
                attn_mask = attn_mask_local
            attn_mask = attn_mask.unsqueeze(1)

        q = self.q(patch_windows).reshape(B * num_windows, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        k_local = self.k_local(patch_windows).reshape(B * num_windows, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        v_local = self.v_local(patch_windows).reshape(B * num_windows, self.window_size * self.window_size, self.num_heads, C // self.num_heads)

        k_list = [k_local]
        v_list = [v_local]
        if global_tokens is not None:
            global_kv = global_tokens[:, None].expand(B, num_windows, -1, -1).reshape(B * num_windows, -1, C)
            k_global = self.k_global(global_kv).reshape(B * num_windows, global_tokens.shape[1], self.num_heads, C // self.num_heads)
            v_global = self.v_global(global_kv).reshape(B * num_windows, global_tokens.shape[1], self.num_heads, C // self.num_heads)
            k_list.append(k_global)
            v_list.append(v_global)

        q = q.transpose(1, 2)
        k = torch.cat(k_list, dim=1).transpose(1, 2)
        v = torch.cat(v_list, dim=1).transpose(1, 2)

        if rope_windows is not None:
            q, k = self.apply_rope(q, k, rope_windows)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2)
        x = x.reshape(B * num_windows, self.window_size * self.window_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        merged = window_reverse(x, self.window_size, H, W, B)
        if self.shift_size > 0:
            merged = torch.roll(merged, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        merged = merged.view(B, H * W, C)
        return merged

