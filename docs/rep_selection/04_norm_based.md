# Norm-Based Representative Token Selection

## Purpose

Select the token with the largest feature norm.

This is the simplest content-aware representative token selection method.

---

## Formula

For each token:

$$
s_i=\|z_i\|_2^2
$$

Select:

$$
i^*=\arg\max_i s_i
$$

Gather:

$$
r_w=g_{i^*}
$$

Use squared norm to avoid square root.

---

## Output

```python
selected_index: [B, num_windows]
rep_vector: [B, num_windows, d]
rep_position: Optional[B, num_windows, 2]
score: Optional[B, num_windows, n]
```

---

## Complexity

$$
\mathcal{O}(nd)
$$

Only one reduction over the channel dimension is required.

---

## Interpretation

This method assumes that a token with larger activation magnitude carries stronger information.

---

## Important Note for Post-LayerNorm

LayerNorm tends to make token norms similar.

If LayerNorm has no affine parameters, or if \(\gamma\) is close to constant and \(\beta\) is small, then:

$$
\|\mathrm{LN}(x_i)\|_2 \approx \sqrt{d}
$$

Therefore, norm-based selection after LayerNorm may become nearly degenerate.

This is an important ablation:

```text
norm-based pre-LN may be meaningful
norm-based post-LN may collapse to almost arbitrary selection
```

---

## Pre-LN vs Post-LN

Pre-LN:

$$
i^*=\arg\max_i\|x_i^{\mathrm{pre}}\|_2^2
$$

Post-LN:

$$
i^*=\arg\max_i\|x_i^{\mathrm{post}}\|_2^2
$$
