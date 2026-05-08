# Window Representative Token Selection Guidelines

This folder contains markdown instructions for implementing linear-time window representative token/position selection policies.

## Files

- `00_common_definitions.md`: shared notation, tensor shape, LayerNorm experiment design, masking, and complexity constraints
- `01_mean_centroid.md`
- `02_centroid_proximal_approx_medoid.md`
- `03_deviation_from_mean.md`
- `04_norm_based.md`
- `05_finite_difference_saliency.md`
- `06_weighted_quadrature_pooling.md`
- `07_weighted_position_estimation.md`
- `08_haar_ll.md`
- `09_entropy_based.md`

## Claude Code Prompt

```md
Implement linear-time window representative token selection policies in PyTorch.

Read the markdown files in `docs/rep_selection/`.

Requirements:

1. Input tensors:
   - `X_pre`: shape `[B, num_windows, n, d]`
   - `X_post`: shape `[B, num_windows, n, d]`
   - optional `valid_mask`: shape `[B, num_windows, n]`
   - optional `positions`: shape `[n, 2]`

2. Implement the following policies:
   - mean_centroid
   - centroid_proximal
   - deviation_from_mean
   - norm_based
   - finite_difference_saliency
   - weighted_quadrature_pooling
   - weighted_position_estimation
   - haar_ll
   - entropy_based

3. Every method must support:
   - `selector_space in {"pre_ln", "post_ln"}`
   - `gather_space in {"pre_ln", "post_ln"}`

4. Every method must have linear complexity:
   \(O(nd)\) per window.

5. Do not implement:
   - pairwise distance matrix
   - exact medoid
   - SVD/PCA
   - attention recomputation
   - graph centrality

6. Return a dictionary with:
   - `rep_vector`
   - `selected_index` if applicable
   - `rep_position` if applicable
   - `score` if applicable

7. Use deterministic tie-breaking by selecting the smallest index.

8. Use squared L2 norms instead of L2 norms when possible to avoid unnecessary square roots.
```
