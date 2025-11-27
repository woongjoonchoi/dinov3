# Baseline 3 Codex Prompt

Use the following prompt verbatim in Codex. After the two placeholders at the end, paste the full source for the original global DINOv3 ViT backbone and the window-only backbone.

````text
Act as a senior PyTorch architect.

I am working with DINOv3-style ViT backbones and already have:

1. The **original global DINOv3 Vision Transformer** implementation
   (all blocks use full global self-attention over `[CLS + storage + patches]`).

2. A **window-only variant** that uses my custom window attention / OWA kernel
   for the patch tokens (including shifted-window attention and its attention masks).

Baseline 2 (CLS removed + pure window + GAP-only) causes DINOv3 performance to collapse,
because the pretrained model relies heavily on:

- full global self-attention at every layer, and
- `CLS + storage` tokens acting as global memory / registers.

So I now want to implement a **better hybrid** called **Baseline 3** in a new directory:

- `dinov3_window_base1_3/`

The key idea:

> Treat `CLS + storage` as **global tokens**, and in **every block** run:
>
> - a **local (shifted-window) attention branch** on patch tokens, where patches attend to:
>   - other patches in the same window (with shifted-window masking), and
>   - the global tokens as **additional K/V** (unmasked);
> - a **global branch** where the global tokens (CLS+storage) attend to all tokens (global view).

This should behave like a Vision Longformer / GCViT-style local+global hybrid, but tailored for DINOv3:

- global tokens are **never windowed or duplicated**,
- patch tokens are windowed and shifted as in the window-only variant,
- CLS + storage maintain their semantics as “global memory” across layers.

---

### Token layout and split

Each block receives a sequence:

- `x` of shape `[B, 1 + n_storage + N, C]`:

  ```python
  G = 1 + n_storage  # number of global tokens (CLS + storage)

  global_tokens = x[:, :G, :]   # [B, G, C], CLS + storage = global tokens
  patch_tokens  = x[:, G:, :]   # [B, N, C], image patches only
  ```

We will use:

* `global_tokens` as a **single global token set per image** (never window-partitioned),
* `patch_tokens` as the local tokens that will be window-partitioned.

---

### Local branch: patch Q → (local patches + global tokens K/V)

1. **Window partition (shifted-window) on patch tokens only**

   Use the same window partitioning and **shifted-window logic** from the existing window-only backbone:

   ```python
   # patch_tokens: [B, N, C] with spatial shape (H, W)
   patch_windows = window_partition(patch_tokens, window_size)  # [B * nW, Ws*Ws, C]
   ```

   * Important: **do NOT include `global_tokens` in window_partition**.
   * `nW`: number of windows, `Ws`: window size (e.g., 14), so `Ws*Ws` tokens per window.

2. **Broadcast global tokens as K/V to each window**

   Each window should be able to attend to the same set of global tokens.
   Replicate `global_tokens` over windows:

   ```python
   # global_tokens: [B, G, C]
   global_kv = (
       global_tokens[:, None]              # [B, 1, G, C]
                  .expand(B, nW, G, C)     # [B, nW, G, C]
                  .reshape(B * nW, G, C)   # [B*nW, G, C]
   )
   ```

3. **Compute Q/K/V**

   Patches query both local patches and global tokens:

   ```python
   Q_patch = Wq(patch_windows)            # [B*nW, Ws*Ws, Cq]

   K_local = Wk(patch_windows)            # [B*nW, Ws*Ws, Ck]
   V_local = Wv(patch_windows)

   K_global = Wk_g(global_kv)             # [B*nW, G, Ck]
   V_global = Wv_g(global_kv)

   K_all = torch.cat([K_local, K_global], dim=1)  # [B*nW, Ws*Ws + G, Ck]
   V_all = torch.cat([V_local, V_global], dim=1)
   ```

   * Q is **only from patch tokens**.
   * Global tokens are used **only as K/V** in this branch.
   * This ensures we do **not** create per-window CLS outputs.

4. **Shifted-window attention mask extended for global tokens**

   Assume the existing shifted-window mask for patches is:

   ```python
   # [nW, Ws*Ws, Ws*Ws], used to mask attention across different windows
   attn_mask_local
   ```

   Extend it to allow all patch→global attention to pass unmasked:

   ```python
   zeros_for_global = torch.zeros(
       nW, Ws*Ws, G,
       device=attn_mask_local.device,
       dtype=attn_mask_local.dtype,
   )
   # Now each row (patch token) has:
   #   - standard mask for other patches (Ws*Ws entries)
   #   - zeros for all global tokens (G entries)
   attn_mask = torch.cat(
       [attn_mask_local, zeros_for_global], dim=-1
   )  # [nW, Ws*Ws, Ws*Ws + G]
   ```

   * Patch↔patch: same shifted-window mask as Swin (different windows masked out with large negative values).
   * Patch↔global: mask value 0 → all patches can see all global tokens.

   Call the attention:

   ```python
   patch_windows_out = attention(
       Q_patch, K_all, V_all,
       attn_mask=attn_mask,
       ...
   )  # [B*nW, Ws*Ws, C]
   ```

5. **Reverse windows back to patch_tokens_out**

   ```python
   patch_tokens_out = window_reverse(
       patch_windows_out, window_size, H, W
   )  # [B, N, C]
   ```

   * Only patches go through `window_reverse`.
   * `global_tokens` are **untouched** by this local branch.

---

### Global branch: update CLS + storage via full global attention

To keep `CLS + storage` as global memory, we add a **global branch** that updates them
with full global attention:

1. Concatenate current global and (optionally updated) patch tokens:

   ```python
   # Option 1: use locally updated patches
   x_full = torch.cat([global_tokens, patch_tokens_out], dim=1)  # [B, G + N, C]

   # Option 2 (simpler): use original patch_tokens instead
   # x_full = torch.cat([global_tokens, patch_tokens], dim=1)
   ```

2. Compute global attention where:

   * Query = global tokens only (`CLS + storage`),
   * Key/Value = all tokens (`global + patch`):

   ```python
   Q_g = Wq_global(x_full[:, :G, :])   # [B, G, Cq]
   K_g = Wk_global(x_full)             # [B, G + N, Ck]
   V_g = Wv_global(x_full)             # [B, G + N, Cv]

   global_tokens_out = attention(
       Q_g, K_g, V_g,
       attn_mask=None,   # full global attention, no window masks
       ...
   )  # [B, G, C]
   ```

   * No shifted-window mask is applied here.
   * Global tokens fully see all patches and themselves, preserving DINOv3 semantics.

3. For a simpler initial version, you **do not need an extra patch→global cross-attention**
   because patches already saw global tokens in the local branch via K/V.
   (If you want, you can later add a separate `patch_to_global_attention` step.)

---

### Block output and residual structure

Put everything into a `LocalGlobalBlock` that follows the same pattern as the original DINOv3 `Block`:

* Input: `x` (`[B, G+N, C]`).

* Split into `global_tokens` and `patch_tokens`.

* Apply norm + attention + residual as usual, but implement the attention part as:

  1. local branch on patches (window + global K/V),
  2. global branch on `global_tokens` (global Q over all tokens),
  3. recombine the outputs:

     ```python
     x_attn = torch.cat([global_tokens_out, patch_tokens_out], dim=1)
     x = x + x_attn  # standard residual
     x = x + mlp(norm2(x))  # same MLP + residual as original Block
     ```

* Ensure that:

  * **global tokens are never window-partitioned**,
  * **window masks are only applied to patch tokens**, 
  * **no per-window CLS is ever created** (no duplication of global tokens as Q),
  * `window_reverse` is applied only on patch tokens.

---

### Backbone class and API

Create a new package:

* `dinov3_window_base1_3/__init__.py`
* `dinov3_window_base1_3/vit.py`

In `vit.py`:

* Define a new backbone class, for example:

  * `Baseline3LocalGlobalVisionTransformer`
    or
  * `LocalGlobalHybridVisionTransformer`

* Implementation requirements:

  1. **Constructor**

     * Accept the same arguments as the existing DINOv3 ViT/window backbones
       (embed_dim, depth, num_heads, mlp_ratio, etc.).
     * Use the same patch embedding and CLS/storage token handling as the original global model.

  2. **Blocks**

     * Replace the standard `Block` stack with `depth` instances of `LocalGlobalBlock`.
     * Each `LocalGlobalBlock` implements the behavior described above.

  3. **Forward API**

     * `forward_features` / `forward_features_list` should behave like the original DINOv3 code:

       * prepend CLS and storage tokens,
       * feed through all `LocalGlobalBlock`s,
       * apply final norm,
       * return a dictionary with the same keys:

         ```python
         {
             "x_norm_clstoken":   cls_token,        # [B, C]
             "x_storage_tokens":  storage_tokens,   # [B, G-1, C] or empty
             "x_norm_patchtokens": patch_tokens,    # [B, N, C]
             ...
         }
         ```

     * This ensures existing heads (DINO loss head, ImageNet linear head, etc.)
       continue to work without modification.

---

### Shifted-window mask requirements (very important)

* Reuse the **shifted-window mask construction** from the existing window-only implementation.

* When using the shifted-window mask:

  * Apply it **only** to patch tokens.
  * Global tokens are:

    * **not window-partitioned**, and
    * **never masked out** in any attention operation.

* Concretely, for the local branch:

  * Extend the patch-only `attn_mask_local` (shape `[nW, Ws*Ws, Ws*Ws]`) to include G extra K positions for global tokens, with all zeros in those extra columns.

* For the global branch:

  * Perform full attention with **no window mask** (CLS+storage query all tokens).

---

### Weight reuse and loading

* Reuse as many modules and parameter definitions as possible from:

  * the original global backbone (patch embed, CLS/storage parameters, norms, MLPs, etc.),
  * the window-only backbone (window partition/reverse, shifted-window masks, OWA integration, RoPE).

* For new attention modules (e.g., `Wq_global`, `Wk_g`, `Wv_g`), initialize them properly
  and allow `load_state_dict` to use `strict=False` when loading pretrained DINOv3 weights.

---

### Public API

* In `dinov3_window_base1_3/__init__.py` export the main class, e.g.:

  ```python
  from .vit import LocalGlobalHybridVisionTransformer
  ```

* The goal is that training/evaluation code can switch to Baseline 3 by simply changing the import:

  ```python
  from dinov3_window_base1_3 import LocalGlobalHybridVisionTransformer
  ```

  without changing the downstream head code.

---

### Output format

* **Do NOT modify any existing files.**

* Output **only** the complete contents for the new files:

  * `dinov3_window_base1_3/__init__.py`
  * `dinov3_window_base1_3/vit.py`

* The code must be valid Python, ready to be saved into the repository.

---

Below I will paste:

1. The original global DINOv3 ViT implementation file.
2. The current window-only backbone implementation file.

Use them as references and implement Baseline 3 in `dinov3_window_base1_3` exactly as described above.

<<<ORIGINAL GLOBAL VIT IMPLEMENTATION BELOW>>>

<<<WINDOW-ONLY VIT IMPLEMENTATION BELOW>>>

```
