# Centroid-Proximal / Approximate Medoid Selection

## Purpose

Select the actual patch token closest to the window centroid.

This is a linear-time approximation of exact medoid selection.

Exact medoid is:

$$
i^* = \arg\min_i \sum_{j=1}^{n}\|z_i-z_j\|_2
$$

but exact medoid requires pairwise distances and costs:

$$
\mathcal{O}(n^2d)
$$

This method avoids pairwise distances by using the centroid.

---

## Formula

First compute the window centroid:

$$
\mu_w=\frac{1}{n}\sum_{i=1}^{n}z_i
$$

With valid mask:

$$
\mu_w=\frac{\sum_i m_i z_i}{\sum_i m_i+\epsilon}
$$

Then select the token closest to the centroid:

$$
i^*=\arg\min_i\|z_i-\mu_w\|_2^2
$$

Use squared distance to avoid unnecessary square root.

The representative token is gathered from the gather feature:

$$
r_w=g_{i^*}
$$

where \(g_i\) comes from `gather_space`.

---

## Output

```python
selected_index: [B, num_windows]
rep_vector: [B, num_windows, d]
rep_position: Optional[B, num_windows, 2]
```

---

## Complexity

Centroid computation:

$$
\mathcal{O}(nd)
$$

Distance-to-centroid computation:

$$
\mathcal{O}(nd)
$$

Total:

$$
\mathcal{O}(nd)
$$

---

## Interpretation

This selects the most central actual token in the window.

It is useful when the representative should preserve a real spatial position instead of creating a synthetic mean vector.

---

## Pre-LN vs Post-LN

Pre-LN version:

$$
i^*=\arg\min_i\|x_i^{\mathrm{pre}}-\mu_w^{\mathrm{pre}}\|_2^2
$$

Post-LN version:

$$
i^*=\arg\min_i\|x_i^{\mathrm{post}}-\mu_w^{\mathrm{post}}\|_2^2
$$

Pre-LN selection is affected by token magnitude.

Post-LN selection focuses more on normalized feature direction.
