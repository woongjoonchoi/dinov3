# Finite-Difference Saliency-Based Selection

## Purpose

Select the token with the largest local feature variation.

This method treats window tokens as a 2D spatial feature grid and uses finite differences to estimate local saliency.

---

## Input Shape

For this method, reshape each window into a grid:

```python
Z_grid: [B, num_windows, M, M, d]
```

where:

$$
n=M^2
$$

---

## Formula: 4-Neighbor Local Contrast

For token at local coordinate \((u,v)\), define its 4-neighbor set:

$$
\mathcal{N}(u,v)
=
\{(u-1,v),(u+1,v),(u,v-1),(u,v+1)\}
$$

using only valid in-bound neighbors.

The saliency score is:

$$
s_{u,v}
=
\sum_{(a,b)\in\mathcal{N}(u,v)}
\|z_{u,v}-z_{a,b}\|_2^2
$$

Optionally normalize by the number of valid neighbors:

$$
s_{u,v}
=
\frac{1}{|\mathcal{N}(u,v)|}
\sum_{(a,b)\in\mathcal{N}(u,v)}
\|z_{u,v}-z_{a,b}\|_2^2
$$

Select:

$$
(u^*,v^*)=\arg\max_{u,v}s_{u,v}
$$

Convert to local flat index:

$$
i^*=u^*M+v^*
$$

Gather:

$$
r_w=g_{i^*}
$$

---

## Alternative Formula: Forward Difference

Another valid implementation is:

$$
D_x z_{u,v}=z_{u+1,v}-z_{u,v}
$$

$$
D_y z_{u,v}=z_{u,v+1}-z_{u,v}
$$

$$
s_{u,v}
=
\|D_x z_{u,v}\|_2^2+\|D_y z_{u,v}\|_2^2
$$

For boundary tokens, use only available differences.

---

## Output

```python
selected_index: [B, num_windows]
rep_vector: [B, num_windows, d]
rep_position: Optional[B, num_windows, 2]
score: Optional[B, num_windows, M, M]
```

---

## Complexity

Each token compares only with a constant number of neighbors.

$$
\mathcal{O}(nd)
$$

Do not compute all pairwise token differences.

---

## Interpretation

This method selects the token with the strongest local feature change.

It may capture:

- edge-like regions
- object boundaries
- local texture changes
- semantic transition regions

---

## Pre-LN vs Post-LN

Pre-LN saliency:

$$
s_{u,v}
=
\sum_{(a,b)\in\mathcal{N}(u,v)}
\|x_{u,v}^{\mathrm{pre}}-x_{a,b}^{\mathrm{pre}}\|_2^2
$$

Post-LN saliency:

$$
s_{u,v}
=
\sum_{(a,b)\in\mathcal{N}(u,v)}
\|x_{u,v}^{\mathrm{post}}-x_{a,b}^{\mathrm{post}}\|_2^2
$$

Pre-LN captures raw magnitude and local variation.

Post-LN captures normalized directional variation.
