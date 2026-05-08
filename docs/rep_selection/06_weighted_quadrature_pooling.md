# Weighted Quadrature Pooling

## Purpose

Create a representative vector as a weighted sum of tokens.

This method does not necessarily select an existing patch token.

It interprets the window representative as a discrete approximation of a local feature integral.

---

## Formula

Given token scores \(s_i\), compute normalized weights:

$$
\alpha_i=
\frac{\exp(s_i/\tau)}
{\sum_{j=1}^{n}\exp(s_j/\tau)}
$$

For numerical stability:

$$
\alpha_i=
\frac{\exp((s_i-\max_j s_j)/\tau)}
{\sum_{j=1}^{n}\exp((s_j-\max_j s_j)/\tau)}
$$

The representative vector is:

$$
r_w=\sum_{i=1}^{n}\alpha_i g_i
$$

where \(g_i\) comes from `gather_space`.

---

## Score Sources

The score \(s_i\) must come from a linear-time score function, for example:

### Norm score

$$
s_i=\|z_i\|_2^2
$$

### Deviation score

$$
s_i=\|z_i-\mu_w\|_2^2
$$

### Saliency score

$$
s_i=\|\nabla z_i\|_2^2
$$

### Entropy score

$$
s_i=H(z_i)
$$

Do not use pairwise or attention-based scores.

---

## Output

```python
rep_vector: [B, num_windows, d]
weights: Optional[B, num_windows, n]
selected_index: Optional[B, num_windows]
```

If an actual token index is needed, use:

$$
i^*=\arg\max_i \alpha_i
$$

or:

$$
i^*=\arg\min_i\|g_i-r_w\|_2^2
$$

The first option is cheaper.

---

## Complexity

If the score function is linear:

$$
\mathcal{O}(nd)
$$

Weighted sum:

$$
\mathcal{O}(nd)
$$

Total:

$$
\mathcal{O}(nd)
$$

---

## Interpretation

This method is a soft representative selection method.

It is useful when hard token selection is too unstable.
