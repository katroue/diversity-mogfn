# Extended Pareto Front Smoothness (PFS) Implementation

## Overview

The PFS metric has been extended to support multi-objective problems with 3+ objectives.

## What Changed

### Original Implementation (2D Only)
- **Method**: 1D curve fitting on Pareto front
- **Limitations**:
  - Only worked for 2-objective problems
  - For 3+ objectives, silently ignored extra dimensions
  - Only analyzed first two objectives

### New Implementation (2+ Objectives)
- **For 2 objectives**: Uses original 1D curve fitting (backward compatible)
- **For 3+ objectives**:
  1. Projects Pareto front to 2D manifold using PCA
  2. Applies curve fitting to projected points
  3. Falls back to k-NN variance if PCA fails or explains <50% variance

## Technical Details

### Algorithm

```
IF num_objectives == 2:
    Use original curve fitting
ELSE IF num_objectives >= 3:
    1. Extract Pareto front
    2. Apply PCA to project to 2D manifold
    3. Check explained variance ratio
       - IF explained_var >= 0.5:
           Apply curve fitting to 2D projection
       - ELSE:
           Use k-NN fallback (variance in neighbor distances)
```

### Key Functions

- `pareto_front_smoothness()` - Main entry point (src/metrics/objective.py)
- `_pfs_2d()` - Original 2D implementation
- `_pfs_multiobjective()` - New 3+ objective implementation
- `_pfs_curve_fitting()` - Shared curve fitting logic
- `_pfs_knn_fallback()` - Fallback for degenerate cases

## Recalculating Existing Results

Use the provided script to update all experiments:

```bash
# Dry run (preview changes)
python scripts/metrics/recalculate_pfs.py \
    --results_dir results/ablations/capacity \
    --dry_run

# Actually update
python scripts/metrics/recalculate_pfs.py \
    --results_dir results/ablations/capacity

# Update all ablations
python scripts/metrics/recalculate_pfs.py \
    --results_dir results/ablations
```

### What the Script Does

1. Finds all experiments with `objectives.npy` files
2. Loads existing `metrics.json`
3. Recalculates PFS using extended implementation
4. **Backs up original metrics** to `metrics_backup.json`
5. Updates `metrics.json` with new PFS value
6. Adds metadata about recalculation

### Backup Safety

- Original metrics saved to `metrics_backup.json` (one-time backup)
- Metadata added to track recalculation:
  ```json
  {
    "pfs": 0.015274,
    "pfs_metadata": {
      "recalculated": true,
      "recalculation_date": "2025-12-01T20:46:21",
      "old_value": 0.000000,
      "method": "extended_pfs_with_manifold_projection"
    }
  }
  ```

## Expected Changes

### For 2-Objective Problems (HyperGrid)
- **Most experiments**: PFS changes from 0.0 to small positive values (0.001-0.03)
- **Why**: Original implementation had many edge cases returning 0.0
- **Interpretation**: Lower PFS still = smoother front

### For 3-Objective Problems (DNA Sequences)
- **All experiments**: PFS now measures full 3D Pareto surface (via 2D projection)
- **Before**: Only measured 2D slice (first two objectives)
- **After**: Captures full manifold structure

## Dependencies

The extended implementation requires:
- `sklearn` (for PCA and k-NN)
- `numpy`
- `scipy` (for distance metrics)

These are already in `requirements.txt`.

## Validation

Test the implementation:

```python
import numpy as np
from src.metrics.objective import pareto_front_smoothness

# 2D test
obj_2d = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]])
pfs_2d = pareto_front_smoothness(obj_2d)
print(f"2D PFS: {pfs_2d}")

# 3D test
obj_3d = np.array([
    [0.1, 0.8, 0.7],
    [0.3, 0.6, 0.8],
    [0.5, 0.4, 0.6],
    [0.7, 0.2, 0.4],
    [0.9, 0.1, 0.2],
    [0.2, 0.7, 0.5],
    [0.4, 0.5, 0.7],
    [0.6, 0.3, 0.5],
    [0.8, 0.2, 0.3],
    [0.1, 0.9, 0.8]
])
pfs_3d = pareto_front_smoothness(obj_3d)
print(f"3D PFS: {pfs_3d}")
```

## References

- Original PFS concept: Measures deviation from smooth Pareto front curve
- Manifold learning: Assumes Pareto fronts lie on low-dimensional manifolds
- PCA projection: Finds principal components capturing most variance

## Notes

- **Backward compatible**: 2-objective behavior unchanged
- **Robust**: Multiple fallback mechanisms for edge cases
- **Interpretable**: Lower PFS = smoother/more continuous Pareto front
- **Efficient**: O(n²) complexity for both 2D and multi-objective cases
