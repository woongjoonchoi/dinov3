"""
Backbone V1: Mean Centroid window representative token.

Preserve & Add: global register tokens unchanged; one win_rep token per window
is inserted between local patches and global tokens in the K/V sequence.

K = cat([k_local, k_win_reg, k_global], dim=2)
V = cat([v_local, v_win_reg, v_global], dim=2)
"""
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SwiGLUFFN
from dinov3.layers.attention import rope_apply
from dinov3.utils import named_apply
from dionv3_window.layers.window_attention import window_partition, window_reverse

from .rep_selection.selectors import select_v1_mean_centroid


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


class WindowSelfAttentionWithGlobalV1(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        shift_size: int = 0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
        selector_space: str = "pre_ln",
        gather_space: str = "post_ln",
        selector_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.selector_space = selector_space
        self.gather_space = gather_space
        self.selector_kwargs = selector_kwargs or {}

        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)
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
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        if sin.dim() == 3:
            sin = sin.unsqueeze(1)
            cos = cos.unsqueeze(1)
        rope_dtype = sin.dtype
        q, k = q.to(dtype=rope_dtype), k.to(dtype=rope_dtype)
        N, sin_len = q.shape[-2], sin.shape[-2]
        if sin_len > N:
            sin, cos = sin[:, :, sin_len - N:, :], cos[:, :, sin_len - N:, :]
            sin_len = N
        prefix = N - sin_len
        q = torch.cat((q[:, :, :prefix, :], rope_apply(q[:, :, prefix:, :], sin, cos)), dim=-2)
        k = torch.cat((k[:, :, :prefix, :], rope_apply(k[:, :, prefix:, :], sin, cos)), dim=-2)
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def forward(
        self,
        patch_tokens: Tensor,
        hw: Tuple[int, int],
        rope: Tensor | Tuple[Tensor, Tensor] | None = None,
        global_tokens: Tensor | None = None,
        *,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        patch_tokens_pre: Tensor | None = None,
    ) -> Tensor:
        B, N, C = patch_tokens.shape
        H, W = hw
        assert N == H * W

        # Window-partition post-LN tokens
        patch_view = patch_tokens.view(B, H, W, C)
        if self.shift_size > 0:
            shifted = torch.roll(patch_view, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = patch_view

        # Window-partition pre-LN tokens (for selector)
        if patch_tokens_pre is not None:
            pre_view = patch_tokens_pre.view(B, H, W, C)
            if self.shift_size > 0:
                pre_shifted = torch.roll(pre_view, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                pre_shifted = pre_view
            patch_windows_pre = window_partition(pre_shifted, self.window_size)
        else:
            patch_windows_pre = None

        # Build RoPE windows
        rope_windows = None
        if rope is not None:
            sin, cos = rope
            sin = sin.view(1, H, W, -1)
            cos = cos.view(1, H, W, -1)
            if self.shift_size > 0:
                sin = torch.roll(sin, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                cos = torch.roll(cos, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            sin_w = window_partition(sin, self.window_size).view(-1, self.window_size * self.window_size, sin.shape[-1])
            cos_w = window_partition(cos, self.window_size).view(-1, self.window_size * self.window_size, cos.shape[-1])
            num_w_rope = sin_w.shape[0]
            sin_w = sin_w.expand(B, num_w_rope, -1, -1).reshape(B * num_w_rope, self.window_size * self.window_size, sin.shape[-1])
            cos_w = cos_w.expand(B, num_w_rope, -1, -1).reshape(B * num_w_rope, self.window_size * self.window_size, cos.shape[-1])
            rope_windows = (sin_w, cos_w)

        patch_windows = window_partition(shifted, self.window_size)  # [B*W, n, C]
        num_windows = patch_windows.shape[0] // B
        n = self.window_size * self.window_size

        # ── Compute window representative token (V1: mean centroid) ──────────
        X_post = patch_windows.reshape(B, num_windows, n, C)
        X_pre = patch_windows_pre.reshape(B, num_windows, n, C) if patch_windows_pre is not None else X_post
        rep_dict = select_v1_mean_centroid(
            X_pre, X_post,
            selector_space=self.selector_space,
            gather_space=self.gather_space,
            **self.selector_kwargs,
        )
        rep_flat = rep_dict["rep_vector"].reshape(B * num_windows, 1, C)  # [B*W, 1, C]

        # ── Attention mask (shift case): insert zeros for win_reg slot ────────
        attn_mask = None
        if self.shift_size > 0:
            attn_mask_local = self.get_attention_mask(H, W, patch_tokens.device, patch_tokens.dtype, B)
            zeros_win_reg = torch.zeros(
                attn_mask_local.shape[0], attn_mask_local.shape[1], 1,
                device=attn_mask_local.device, dtype=attn_mask_local.dtype,
            )
            if global_tokens is not None:
                zeros_global = torch.zeros(
                    attn_mask_local.shape[0], attn_mask_local.shape[1], global_tokens.shape[1],
                    device=attn_mask_local.device, dtype=attn_mask_local.dtype,
                )
                attn_mask = torch.cat([attn_mask_local, zeros_win_reg, zeros_global], dim=-1)
            else:
                attn_mask = torch.cat([attn_mask_local, zeros_win_reg], dim=-1)
            attn_mask = attn_mask.unsqueeze(1)

        # ── Q / K_local / V_local projections ────────────────────────────────
        q = q_proj(patch_windows).reshape(B * num_windows, n, self.num_heads, C // self.num_heads)
        k_local = k_proj(patch_windows).reshape(B * num_windows, n, self.num_heads, C // self.num_heads)
        v_local = v_proj(patch_windows).reshape(B * num_windows, n, self.num_heads, C // self.num_heads)

        k_list = [k_local]
        v_list = [v_local]

        if global_tokens is not None:
            global_kv = global_tokens[:, None].expand(B, num_windows, -1, -1).reshape(B * num_windows, -1, C)
            k_global = k_proj(global_kv).reshape(B * num_windows, global_tokens.shape[1], self.num_heads, C // self.num_heads)
            v_global = v_proj(global_kv).reshape(B * num_windows, global_tokens.shape[1], self.num_heads, C // self.num_heads)
            k_list.append(k_global)
            v_list.append(v_global)

        q = q.transpose(1, 2)
        k_local = k_local.transpose(1, 2)
        v_local = v_local.transpose(1, 2)

        if rope_windows is not None:
            q, k_local = self.apply_rope(q, k_local, rope_windows)

        # ── Window representative K / V ───────────────────────────────────────
        k_win_reg = k_proj(rep_flat).reshape(B * num_windows, 1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_win_reg = v_proj(rep_flat).reshape(B * num_windows, 1, self.num_heads, C // self.num_heads).transpose(1, 2)

        # K = [k_local, k_win_reg, k_global], V = [v_local, v_win_reg, v_global]
        k = torch.cat([k_local, k_win_reg] + [kp.transpose(1, 2) for kp in k_list[1:]], dim=2)
        v = torch.cat([v_local, v_win_reg] + [vp.transpose(1, 2) for vp in v_list[1:]], dim=2)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B * num_windows, n, C)
        x = out_proj(x)
        x = self.proj_drop(x)

        merged = window_reverse(x, self.window_size, H, W, B)
        if self.shift_size > 0:
            merged = torch.roll(merged, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return merged.view(B, H * W, C)


class LocalGlobalBlockV1(nn.Module):
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
        selector_space: str = "pre_ln",
        gather_space: str = "post_ln",
        selector_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_global_tokens = num_global_tokens
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias, device=device)

        self.local_attn = WindowSelfAttentionWithGlobalV1(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
            selector_space=selector_space,
            gather_space=gather_space,
            selector_kwargs=selector_kwargs,
        )

        self.enable_patch_to_global = enable_patch_to_global
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
        q = self.q_proj(q_tokens).reshape(B, G, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(kv_tokens).reshape(B, kv_tokens.shape[1], self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(kv_tokens).reshape(B, kv_tokens.shape[1], self.num_heads, -1).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, G, -1)
        out = self.out_proj(out)
        return self.proj_drop(out)

    def _forward_single(self, x: Tensor, hw: Tuple[int, int], rope=None) -> Tensor:
        B, N, _ = x.shape
        assert N >= self.num_global_tokens
        x_norm = self.norm1(x)  # post-LN

        global_tokens = x_norm[:, :self.num_global_tokens, :]
        patch_tokens = x_norm[:, self.num_global_tokens:, :]      # post-LN patch
        patch_tokens_pre = x[:, self.num_global_tokens:, :]        # pre-LN patch

        patch_local = self.local_attn(
            patch_tokens,
            hw=hw,
            rope=rope,
            global_tokens=global_tokens,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            out_proj=self.out_proj,
            patch_tokens_pre=patch_tokens_pre,
        )

        kv_tokens = torch.cat((global_tokens, patch_tokens), dim=1)
        global_out = self._global_attention(global_tokens, kv_tokens)

        attn_out = torch.cat((global_out, patch_local), dim=1)
        x_attn = x + self.ls1(attn_out)
        return x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

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
                rope_or_rope_list = [None] * len(x_or_x_list)
            assert isinstance(hw_or_hw_list, list)
            return [self._forward_single(x, hw, r) for x, hw, r in zip(x_or_x_list, hw_or_hw_list, rope_or_rope_list)]
        else:
            raise AssertionError


class LocalGlobalHybridVisionTransformerV1(nn.Module):
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
        window_size: int = 16,
        device: Any | None = None,
        enable_patch_to_global: bool = True,
        selector_space: str = "pre_ln",
        gather_space: str = "post_ln",
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            print(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        print(f"V1 (mean_centroid) window_size={window_size} selector={selector_space}/{gather_space}")
        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_global_tokens = n_storage_tokens + 1

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, flatten_embedding=False,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim, num_heads=num_heads, base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period, max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords, jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype], device=device,
        )
        ffn_layer_cls = ffn_layer_dict[ffn_layer]

        blocks_list = []
        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            blocks_list.append(
                LocalGlobalBlockV1(
                    dim=embed_dim, num_heads=num_heads,
                    num_global_tokens=self.num_global_tokens,
                    window_size=window_size, shift_size=shift_size,
                    ffn_ratio=ffn_ratio, qkv_bias=qkv_bias,
                    proj_bias=proj_bias, ffn_bias=ffn_bias,
                    drop=drop_path_rate, attn_drop=0.0,
                    norm_layer=norm_layer_cls, act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls, init_values=layerscale_init,
                    mask_k_bias=mask_k_bias, device=device,
                    enable_patch_to_global=enable_patch_to_global,
                    selector_space=selector_space,
                    gather_space=gather_space,
                )
            )
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer_cls(embed_dim)
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def _normalize_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        def strip_prefix(key: str) -> str:
            for p in ["module.", "backbone.", "model."]:
                if key.startswith(p):
                    return key[len(p):]
            return key
        return {strip_prefix(k): v for k, v in state_dict.items()}

    def _remap_pretrained_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
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
            return q_w, k_w, v_w, q_b, k_b, v_b

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
            remapped[f"{prefix}q_proj.weight"] = q_w.clone()
            remapped[f"{prefix}q_proj.bias"] = q_b.clone()
            remapped[f"{prefix}k_proj.weight"] = k_w.clone()
            remapped[f"{prefix}k_proj.bias"] = k_b.clone()
            remapped[f"{prefix}v_proj.weight"] = v_w.clone()
            remapped[f"{prefix}v_proj.bias"] = v_b.clone()
            if proj_w is not None:
                remapped[f"{prefix}out_proj.weight"] = proj_w
            if proj_b is not None:
                remapped[f"{prefix}out_proj.bias"] = proj_b
            remapped.pop(f"{prefix}attn.qkv.weight", None)
            remapped.pop(f"{prefix}attn.qkv.bias", None)
            remapped.pop(f"{prefix}attn.proj.weight", None)
            remapped.pop(f"{prefix}attn.proj.bias", None)
        return remapped

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        remapped = self._remap_pretrained_state_dict(state_dict)
        model_state = self.state_dict()
        filtered = {k: remapped[k] if k in remapped and isinstance(remapped[k], torch.Tensor) else v
                    for k, v in model_state.items()}
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
        storage_tokens = self.storage_tokens if self.n_storage_tokens > 0 else torch.empty(
            1, 0, cls_token.shape[-1], dtype=cls_token.dtype, device=cls_token.device)
        x = torch.cat([cls_token.expand(B, -1, -1), storage_tokens.expand(B, -1, -1), x], dim=1)
        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x, hw_list = [], []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            hw_list.append(hw_tuple)
        for blk in self.blocks:
            rope_sincos = [self.rope_embed(H=H, W=W) for H, W in hw_list]
            x = blk(x, rope_sincos, hw_list)
        output = []
        for idx, (xi, masks) in enumerate(zip(x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(xi[:, :self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(xi[:, :self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(xi[:, :self.n_storage_tokens + 1])
                x_norm_patch = self.norm(xi[:, self.n_storage_tokens + 1:])
            else:
                x_norm = self.norm(xi)
                x_norm_cls_reg = x_norm[:, :self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1:]
            output.append({
                "x_norm_clstoken": x_norm_cls_reg[:, 0],
                "x_storage_tokens": x_norm_cls_reg[:, 1:],
                "x_norm_patchtokens": x_norm_patch,
                "x_prenorm": xi,
                "masks": masks,
            })
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None):
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
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
        assert len(output) == len(blocks_to_take)
        return output

    def get_intermediate_layers(
        self, x: torch.Tensor, *, n: Union[int, Sequence] = 1,
        reshape: bool = False, return_class_token: bool = False,
        return_extra_tokens: bool = False, norm: bool = True,
    ) -> Tuple:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [
                torch.cat((self.cls_norm(o[:, :self.n_storage_tokens+1]), self.norm(o[:, self.n_storage_tokens+1:])), dim=1)
                if self.untie_cls_and_patch_norms else self.norm(o)
                for o in outputs
            ]
        class_tokens = [o[:, 0] for o in outputs]
        extra_tokens = [o[:, 1:self.n_storage_tokens + 1] for o in outputs]
        outputs = [o[:, self.n_storage_tokens + 1:] for o in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [o.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous() for o in outputs]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, x: torch.Tensor | List[torch.Tensor], masks: Optional[torch.Tensor] = None):
        return self.forward_features(x, masks)


__all__ = ["LocalGlobalHybridVisionTransformerV1"]
