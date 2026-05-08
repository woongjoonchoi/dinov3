# Weighted Representative Position Estimation

## Purpose

Estimate a representative spatial position for each window.

This method returns a continuous position rather than necessarily selecting a discrete token.

---

## Inputs

Each token has a local position:

$$
p_i=(u_i,v_i)
$$

where \(p_i\in\mathbb{R}^2\).

Example normalized coordinates:

$$
u_i,v_i\in[0,1]
$$

or integer coordinates:

$$
u_i,v_i\in\{0,1,\dots,M-1\}
$$

---

## Formula

Given linear-time token scores \(s_i\), compute weights:

$$
\alpha_i=
\frac{\exp(s_i/\tau)}
{\sum_{j=1}^{n}\exp(s_j/\tau)}
$$

Then estimate the representative position:

$$
p_w=\sum_{i=1}^{n}\alpha_i p_i
$$

Equivalently:

$$
p_w=
\frac{\sum_i \tilde{s}_i p_i}
{\sum_i \tilde{s}_i+\epsilon}
$$

where \(\tilde{s}_i\) is a nonnegative score.

---

## Output

```python
rep_position: [B, num_windows, 2]
weights: Optional[B, num_windows, n]
selected_index: Optional[B, num_windows]
```

If a discrete token is required, select the nearest token position:

$$
i^*=\arg\min_i\|p_i-p_w\|_2^2
$$

Then gather:

$$
r_w=g_{i^*}
$$

---

## Complexity

Position aggregation alone:

$$
\mathcal{O}(n)
$$

If feature scores are computed from \(d\)-dimensional tokens:

$$
\mathcal{O}(nd)
$$

---

## Interpretation

This method is useful when the representative should be a location rather than a feature vector.

It can be used for level-wise or hierarchical aggregation of window representative positions.
