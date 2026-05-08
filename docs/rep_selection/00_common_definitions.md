# Common Definitions for Window Representative Token Selection

## Goal

Implement linear-time representative token/position selection methods for each local window.

Each method must support experiments where the selection score is computed either:

1. before LayerNorm: `selector_space = "pre_ln"`
2. after LayerNorm: `selector_space = "post_ln"`

The selected representative token can be gathered either from the same feature space or from a different feature space:

- `gather_space = "pre_ln"`
- `gather_space = "post_ln"`

This separation is important because a method may select a token using raw pre-LN features but feed the post-LN selected token into attention.

---

## Tensor Notation

For each window:

$$
X_w=\{x_1,x_2,\dots,x_n\}, \qquad x_i\in\mathbb{R}^d
$$

where:

- \(n=M^2\): number of tokens in a window
- \(M\): window size
- \(d\): embedding dimension

Implementation shape:

```python
X_pre:  [B, num_windows, n, d]
X_post: [B, num_windows, n, d]
valid_mask: Optional[BoolTensor] with shape [B, num_windows, n]
positions: [n, 2]  # local coordinates, e.g. (u, v)
```

LayerNorm is applied per token over the channel dimension:

$$
\mathrm{LN}(x_i)
=
\gamma \odot \frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}+\beta
$$

where:

$$
\mu_i=\frac{1}{d}\sum_{c=1}^{d}x_{i,c}
$$

$$
\sigma_i^2=\frac{1}{d}\sum_{c=1}^{d}(x_{i,c}-\mu_i)^2
$$

---

## Selector Feature

Define:

$$
Z_w =
\begin{cases}
X_w^{\mathrm{pre}}, & \text{if selector\_space = pre\_ln}\\
X_w^{\mathrm{post}}, & \text{if selector\_space = post\_ln}
\end{cases}
$$

All scores are computed from \(Z_w\).

---

## Gather Feature

Define:

$$
G_w =
\begin{cases}
X_w^{\mathrm{pre}}, & \text{if gather\_space = pre\_ln}\\
X_w^{\mathrm{post}}, & \text{if gather\_space = post\_ln}
\end{cases}
$$

If a method returns an index \(i^*\), the representative token is:

$$
r_w = G_{w,i^*}
$$

---

## Complexity Requirement

All methods in this folder must be linear in the number of tokens and channels:

$$
\mathcal{O}(nd)
$$

Do not use:

- pairwise distance matrix: \(\mathcal{O}(n^2d)\)
- SVD/PCA
- attention recomputation
- graph centrality
- exact medoid

---

## Masking Rule

If `valid_mask` is provided:

- invalid tokens must not contribute to means or weighted sums
- invalid tokens must not be selected
- for max-based selection, set invalid scores to \(-\infty\)
- for min-based selection, set invalid scores to \(+\infty\)

---

## Tie Breaking

If multiple tokens have the same score, select the smallest local index.

This makes the result deterministic.

---

## Recommended Experiment Matrix

Run each policy under at least these settings:

```text
1. selector_space = pre_ln,  gather_space = pre_ln
2. selector_space = post_ln, gather_space = post_ln
3. selector_space = pre_ln,  gather_space = post_ln
4. selector_space = post_ln, gather_space = pre_ln
```

The most important setting for pre-norm Transformers is often:

```text
selector_space = pre_ln
gather_space = post_ln
```

because selection uses raw token statistics, but the token fed into attention is normalized.
