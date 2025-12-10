import numpy as np
import torch
from typing import Dict, Any, List
from scipy.spatial.distance import pdist

def quality_diversity_score(objectives: np.ndarray,
                        reference_point: np.ndarray,
                        diversity_metric: str = 'avg_distance',
                        alpha: float = 0.5,
                        normalize: bool = True) -> Dict[str, float]:
    """
    Compute Quality-Diversity Score (QDS).

    QDS balances Pareto quality (hypervolume) and solution diversity.

    QDS = α * normalized_diversity + (1-α) * normalized_hypervolume

    Args:
        objectives: Objective values, shape (N, num_objectives)
        reference_point: Reference point for hypervolume computation
        diversity_metric: Which diversity measure to use:
                        'avg_distance' - average pairwise distance
                        'min_distance' - minimum pairwise distance (PMD)
                        'spread' - range across objectives
        alpha: Weight parameter in [0,1]
            0 = only quality (HV)
            1 = only diversity
            0.5 = equal weight (default)
        normalize: Whether to normalize HV and diversity to [0,1]

    Returns:
        results: Dictionary containing:
                - qds: Combined quality-diversity score
                - hypervolume: Raw hypervolume value
                - diversity: Raw diversity value
                - normalized_hv: Normalized hypervolume (if normalize=True)
                - normalized_div: Normalized diversity (if normalize=True)

    Example:
        >>> objectives = np.array([[0.8, 0.3], [0.6, 0.5], [0.3, 0.8]])
        >>> reference_point = np.array([1.0, 1.0])
        >>> results = quality_diversity_score(objectives, reference_point, alpha=0.5)
        >>> print(f"QDS: {results['qds']:.4f}")
    """
    from scipy.spatial.distance import pdist

    if len(objectives) < 2:
        return {
            'qds': 0.0,
            'hypervolume': 0.0,
            'diversity': 0.0,
            'normalized_hv': 0.0,
            'normalized_div': 0.0
        }

    # ===== 1. Compute Quality (Hypervolume) =====
    hv = _compute_hypervolume_2d(objectives, reference_point)

    # ===== 2. Compute Diversity =====
    if diversity_metric == 'avg_distance':
        # Average pairwise Euclidean distance
        distances = pdist(objectives, metric='euclidean')
        diversity = np.mean(distances)

    elif diversity_metric == 'min_distance':
        # Minimum pairwise distance (PMD)
        distances = pdist(objectives, metric='euclidean')
        diversity = np.min(distances)

    elif diversity_metric == 'spread':
        # Average range across objectives
        ranges = np.max(objectives, axis=0) - np.min(objectives, axis=0)
        diversity = np.mean(ranges)

    else:
        raise ValueError(f"Unknown diversity_metric: {diversity_metric}")

    # ===== 3. Normalize (optional) =====
    if normalize:
        # Normalize hypervolume
        # Max possible HV is when reference point is at origin
        max_hv = np.prod(reference_point)
        normalized_hv = hv / max_hv if max_hv > 0 else 0.0

        # Normalize diversity
        # Max diversity is the diagonal of the objective space
        max_diversity = np.sqrt(np.sum(reference_point ** 2))

        if diversity_metric == 'spread':
            # For spread, max is the reference point range
            max_diversity = np.mean(reference_point)

        normalized_div = diversity / max_diversity if max_diversity > 0 else 0.0

        # Clip to [0, 1] just in case
        normalized_hv = np.clip(normalized_hv, 0, 1)
        normalized_div = np.clip(normalized_div, 0, 1)
    else:
        normalized_hv = hv
        normalized_div = diversity

    # ===== 4. Combine with weight alpha =====
    qds = alpha * normalized_div + (1 - alpha) * normalized_hv

    return {
        'qds': float(qds),
        'hypervolume': float(hv),
        'diversity': float(diversity),
        'normalized_hv': float(normalized_hv),
        'normalized_div': float(normalized_div),
        'alpha': alpha
    }


def _compute_hypervolume_2d(objectives: np.ndarray,
                        reference_point: np.ndarray) -> float:
    """
    Compute 2D hypervolume indicator (for objectives with 2 dimensions).

    For >2 objectives, use pymoo library instead.
    """
    if objectives.shape[1] != 2:
        # Fallback for higher dimensions - use approximation or pymoo
        try:
            from pymoo.indicators.hv import HV
            hv_indicator = HV(ref_point=reference_point)
            return hv_indicator(objectives)
        except ImportError:
            # Simple approximation: sum of products
            return float(np.sum(np.prod(reference_point - objectives, axis=1)))

    # Sort by first objective (ascending)
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_indices]

    # Filter out points dominated by reference point
    valid_mask = np.all(sorted_obj < reference_point, axis=1)
    sorted_obj = sorted_obj[valid_mask]

    if len(sorted_obj) == 0:
        return 0.0

    # Compute 2D hypervolume using standard algorithm
    hv = 0.0
    prev_x = reference_point[0]

    for i in range(len(sorted_obj)):
        x, y = sorted_obj[i]

        # Rectangle: width = (prev_x - x), height = (ref_y - y)
        width = prev_x - x
        height = reference_point[1] - y

        if width > 0 and height > 0:
            hv += width * height

        prev_x = x

    return hv


# Optional: Multi-objective version that returns tradeoff curve
def quality_diversity_tradeoff(objectives: np.ndarray,
                            reference_point: np.ndarray,
                            alphas: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    Compute QDS for multiple alpha values to show quality-diversity tradeoff.

    Args:
        objectives: Objective values
        reference_point: Reference point for HV
        alphas: Array of alpha values (default: [0, 0.1, 0.2, ..., 1.0])

    Returns:
        results: Dictionary with arrays of QDS, HV, diversity for each alpha
    """
    if alphas is None:
        alphas = np.linspace(0, 1, 11)

    qds_values = []
    hv_values = []
    div_values = []

    for alpha in alphas:
        result = quality_diversity_score(objectives, reference_point, alpha=alpha)
        qds_values.append(result['qds'])
        hv_values.append(result['normalized_hv'])
        div_values.append(result['normalized_div'])

    return {
        'alphas': alphas,
        'qds': np.array(qds_values),
        'hypervolume': np.array(hv_values),
        'diversity': np.array(div_values)
    }
