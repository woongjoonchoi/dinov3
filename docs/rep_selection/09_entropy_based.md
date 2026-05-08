# Entropy-Based Representative Token Selection

## Purpose

Select a token based on the entropy of its channel-wise feature distribution.

This method can select either:

1. high-entropy tokens: feature-rich or uncertain tokens
2. low-entropy tokens: sharp or confident tokens

---

## Formula

For each token \(z_i\in\mathbb{R}^d\), convert channel values to a probability distribution:

$$
q_{i,c}
=
\frac{\exp(z_{i,c}/\tau_c)}
{\sum_{k=1}^{d}\exp(z_{i,k}/\tau_c)}
$$

For numerical stability:

$$
q_{i,c}
=
\frac{\exp((z_{i,c}-\max_k z_{i,k})/\tau_c)}
{\sum_{k=1}^{d}\exp((z_{i,k}-\max_k z_{i,k})/\tau_c)}
$$

Entropy:

$$
H_i
=
-\sum_{c=1}^{d}q_{i,c}\log(q_{i,c}+\epsilon)
$$

---

## High-Entropy Selection

$$
i^*=\arg\max_i H_i
$$

This selects the token whose feature response is spread across many channels.

Interpretation:

```text
feature-rich / uncertain / multi-component token
```

---

## Low-Entropy Selection

$$
i^*=\arg\min_i H_i
$$

This selects the token whose feature response is concentrated in a small number of channels.

Interpretation:

```text
sharp / confident / channel-selective token
```

---

## Output

```python
selected_index: [B, num_windows]
rep_vector: [B, num_windows, d]
rep_position: Optional[B, num_windows, 2]
entropy: Optional[B, num_windows, n]
```

---

## Complexity

Channel softmax and entropy per token:

$$
\mathcal{O}(nd)
$$

No pairwise token comparison is required.

---

## Pre-LN vs Post-LN

Pre-LN entropy:

$$
H_i=H(x_i^{\mathrm{pre}})
$$

Post-LN entropy:

$$
H_i=H(x_i^{\mathrm{post}})
$$

Post-LN entropy may be more stable because token-wise scale is normalized.

Pre-LN entropy may reflect both scale and channel distribution.

---

## Implementation Options

Use:

```python
entropy_mode = "high"
```

for:

$$
i^*=\arg\max_i H_i
$$

Use:

```python
entropy_mode = "low"
```

for:

$$
i^*=\arg\min_i H_i
$$
