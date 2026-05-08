# Haar LL / Low-Frequency Representative Extraction

## Purpose

Extract a low-frequency representative from a window using Haar-style local averaging.

This method treats the window as a 2D feature signal.

---

## Input Shape

```python
Z_grid: [B, num_windows, M, M, d]
```

---

## Option A: Global LL Vector

The simplest global low-frequency representative is equivalent to average pooling over the window:

$$
r_w=
\frac{1}{M^2}
\sum_{u=0}^{M-1}
\sum_{v=0}^{M-1}
z_{u,v}
$$

This is a global low-frequency summary.

Complexity:

$$
\mathcal{O}(nd)
$$

However, this is very close to mean pooling.

---

## Option B: Block-Level Haar LL Selection

To make this method different from global mean pooling, compute local \(2\times2\) LL coefficients.

For each \(2\times2\) block:

$$
\ell_{a,b}
=
\frac{1}{4}
\left(
z_{2a,2b}
+
z_{2a+1,2b}
+
z_{2a,2b+1}
+
z_{2a+1,2b+1}
\right)
$$

Define low-frequency block energy:

$$
e_{a,b}=\|\ell_{a,b}\|_2^2
$$

Select the block with the largest low-frequency energy:

$$
(a^*,b^*)=\arg\max_{a,b} e_{a,b}
$$

Then select the actual token inside that block closest to the LL vector:

$$
i^*
=
\arg\min_{i\in\mathrm{block}(a^*,b^*)}
\|z_i-\ell_{a^*,b^*}\|_2^2
$$

Gather:

$$
r_w=g_{i^*}
$$

---

## Handling Odd Window Sizes

If \(M\) is odd, use one of the following:

1. pad the grid to even size using replicate padding
2. ignore the last row/column for block LL
3. fall back to global LL

Use option 1 by default.

---

## Output

```python
rep_vector: [B, num_windows, d]
selected_index: Optional[B, num_windows]
rep_position: Optional[B, num_windows, 2]
```

---

## Complexity

Each token participates in a constant number of additions.

$$
\mathcal{O}(nd)
$$

---

## Interpretation

Haar LL captures low-frequency local structure.

Global LL summarizes the whole window.

Block-level LL selection finds the strongest low-frequency subregion and then selects a real token from it.

---

## Pre-LN vs Post-LN

Pre-LN Haar LL keeps raw magnitude information.

Post-LN Haar LL emphasizes normalized spatial structure.
