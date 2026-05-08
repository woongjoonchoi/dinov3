# Deviation-from-Mean Selection

## Purpose

Select the token that deviates the most from the window mean.

This method identifies a distinctive or outlier-like token in the local window.

---

## Formula

Compute the centroid:

$$
\mu_w=\frac{1}{n}\sum_{i=1}^{n}z_i
$$

With valid mask:

$$
\mu_w=\frac{\sum_i m_i z_i}{\sum_i m_i+\epsilon}
$$

Compute deviation score:

$$
s_i=\|z_i-\mu_w\|_2^2
$$

Select the token with the largest deviation:

$$
i^*=\arg\max_i s_i
$$

Gather representative token:

$$
r_w=g_{i^*}
$$

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

The method requires one pass for the mean and one pass for the deviation score.

---

## Interpretation

This method selects the token that is most different from the local average.

It may capture:

- object boundary
- local anomaly
- foreground token inside background-heavy window
- high-variation semantic token

---

## Pre-LN vs Post-LN

Pre-LN:

$$
s_i=\|x_i^{\mathrm{pre}}-\mu_w^{\mathrm{pre}}\|_2^2
$$

Post-LN:

$$
s_i=\|x_i^{\mathrm{post}}-\mu_w^{\mathrm{post}}\|_2^2
$$

Pre-LN deviation includes raw activation magnitude.

Post-LN deviation is closer to angular or normalized feature deviation.
