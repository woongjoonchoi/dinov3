# Mean / Centroid Representative

## Purpose

Compute one representative vector for each window using the average of all valid tokens.

This method produces a synthetic representative vector, not necessarily an existing patch token.

---

## Formula

Given selector features:

$$
Z_w = \{z_1,z_2,\dots,z_n\}, \qquad z_i\in\mathbb{R}^d
$$

the window centroid is:

$$
\mu_w = \frac{1}{n}\sum_{i=1}^{n}z_i
$$

With a valid mask \(m_i\in\{0,1\}\):

$$
\mu_w =
\frac{\sum_{i=1}^{n}m_i z_i}
{\sum_{i=1}^{n}m_i+\epsilon}
$$

The representative vector is:

$$
r_w = \mu_w
$$

---

## Output

This method returns:

```python
rep_vector: [B, num_windows, d]
selected_index: None
rep_position: None
```

If an actual token index is required, use the centroid-proximal method instead.

---

## Complexity

$$
\mathcal{O}(nd)
$$

Only one reduction over the tokens is required.

---

## Pre-LN vs Post-LN

If `selector_space = "pre_ln"`:

$$
\mu_w = \frac{1}{n}\sum_i x_i^{\mathrm{pre}}
$$

If `selector_space = "post_ln"`:

$$
\mu_w = \frac{1}{n}\sum_i x_i^{\mathrm{post}}
$$

Pre-LN centroid preserves raw magnitude statistics.

Post-LN centroid removes much of the per-token scale variation and focuses more on normalized feature direction.
