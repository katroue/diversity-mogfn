"""
Traditional multi-objective optimization metrics.

These are standard metrics borrowed from the multi-objective optimization
and evolutionary algorithms literature (NSGA-II, MOEA/D, etc.).
"""

import numpy as np
import torch
from typing import Union, Optional, List, Dict
from scipy.spatial.distance import pdist, cdist


def hypervolume(objectives: np.ndarray, 
                reference_point: np.ndarray,
                maximize: bool = False) -> float:
    """
    Compute Hypervolume (HV) indicator.
    
    HV measures the volume of objective space dominated by the Pareto front.
    Higher HV = better Pareto front approximation.
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
        reference_point: Reference point (worst point in each objective)
        maximize: If True, objectives are maximized (default: minimize)
    
    Returns:
        hv: Hypervolume value
    
    Example:
        >>> objectives = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
        >>> reference_point = np.array([1.0, 1.0])
        >>> hv = hypervolume(objectives, reference_point)
        >>> print(f"Hypervolume: {hv:.4f}")
    """
    objectives = np.atleast_2d(objectives)
    
    if maximize:
        # Convert to minimization problem
        objectives = -objectives
        reference_point = -reference_point
    
    # Filter points dominated by reference point
    valid_mask = np.all(objectives <= reference_point, axis=1)
    objectives = objectives[valid_mask]
    
    if len(objectives) == 0:
        return 0.0
    
    num_objectives = objectives.shape[1]
    
    if num_objectives == 2:
        # Use efficient 2D algorithm
        return _hypervolume_2d(objectives, reference_point)
    elif num_objectives == 3:
        # Use efficient 3D algorithm
        return _hypervolume_3d(objectives, reference_point)
    else:
        # Use general algorithm or pymoo
        try:
            from pymoo.indicators.hv import HV
            hv_indicator = HV(ref_point=reference_point)
            return float(hv_indicator(objectives))
        except ImportError:
            # Fallback: approximate using Monte Carlo
            return _hypervolume_monte_carlo(objectives, reference_point)


def _hypervolume_2d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Efficient 2D hypervolume computation."""
    # Sort by first objective
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_indices]
    
    hv = 0.0
    prev_x = reference_point[0]
    
    for i in range(len(sorted_obj)):
        x, y = sorted_obj[i]
        width = prev_x - x
        height = reference_point[1] - y
        
        if width > 0 and height > 0:
            hv += width * height
        
        prev_x = x
    
    return hv


def _hypervolume_3d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Efficient 3D hypervolume computation using WFG algorithm."""
    # Sort by first objective
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_indices]
    
    hv = 0.0
    
    for i in range(len(sorted_obj)):
        # Compute contribution of point i
        point = sorted_obj[i]
        
        # Find the box: from point to reference, excluding dominated parts
        box_min = point
        box_max = reference_point.copy()
        
        # Subtract volumes dominated by later points
        for j in range(i + 1, len(sorted_obj)):
            if np.all(sorted_obj[j] <= box_max):
                # Point j dominates part of the box
                box_max = np.minimum(box_max, sorted_obj[j])
        
        # Compute box volume
        dims = box_max - box_min
        if np.all(dims > 0):
            hv += np.prod(dims)
    
    return hv


def _hypervolume_monte_carlo(objectives: np.ndarray, 
                            reference_point: np.ndarray,
                            n_samples: int = 10000) -> float:
    """Monte Carlo approximation for high-dimensional hypervolume."""
    # Sample random points in the hyperbox
    n_obj = objectives.shape[1]
    ideal_point = np.min(objectives, axis=0)
    
    samples = np.random.uniform(
        low=ideal_point,
        high=reference_point,
        size=(n_samples, n_obj)
    )
    
    # Count points dominated by at least one solution
    dominated_count = 0
    for sample in samples:
        # Check if any objective vector dominates this sample
        if np.any(np.all(objectives <= sample, axis=1)):
            dominated_count += 1
    
    # Approximate hypervolume
    box_volume = np.prod(reference_point - ideal_point)
    hv = (dominated_count / n_samples) * box_volume
    
    return hv


def r2_indicator(objectives: np.ndarray,
                reference_point: np.ndarray,
                weight_vectors: Optional[np.ndarray] = None,
                num_weights: int = 100) -> float:
    """
    Compute R2 indicator (utility-based indicator).
    
    R2 measures the average utility loss compared to the reference point
    across different preference weight vectors.
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
        reference_point: Reference point (typically nadir point)
        weight_vectors: Custom weight vectors (None = generate uniform)
        num_weights: Number of weight vectors if not provided
    
    Returns:
        r2: R2 indicator value (lower is better)
    
    Example:
        >>> objectives = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
        >>> reference_point = np.array([1.0, 1.0])
        >>> r2 = r2_indicator(objectives, reference_point)
        >>> print(f"R2: {r2:.4f}")
    """
    objectives = np.atleast_2d(objectives)
    num_objectives = objectives.shape[1]
    
    # Generate weight vectors if not provided
    if weight_vectors is None:
        if num_objectives == 2:
            # Uniform weights for 2D
            weights = np.linspace(0, 1, num_weights)
            weight_vectors = np.column_stack([weights, 1 - weights])
        else:
            # Generate diverse weight vectors using Dirichlet
            weight_vectors = np.random.dirichlet(
                np.ones(num_objectives), 
                size=num_weights
            )
    
    # Normalize weight vectors
    weight_vectors = weight_vectors / np.sum(weight_vectors, axis=1, keepdims=True)
    
    # Compute R2
    r2_values = []
    
    for weight in weight_vectors:
        # For this weight, find the best utility
        # Utility = weighted Tchebycheff distance to reference point
        utilities = np.max(weight * (objectives - reference_point), axis=1)
        best_utility = np.min(utilities)
        r2_values.append(best_utility)
    
    # R2 is the average over all weights
    r2 = np.mean(r2_values)
    
    return float(r2)


def average_pairwise_distance(objectives: np.ndarray,
                            metric: str = 'euclidean',
                            percentile: Optional[float] = None) -> float:
    """
    Compute average pairwise distance between solutions.
    
    This is the most common diversity metric in the literature.
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
        metric: Distance metric ('euclidean', 'manhattan', 'chebyshev')
        percentile: If provided, return this percentile instead of mean
                (e.g., 50 for median, 10 for 10th percentile)
    
    Returns:
        avg_dist: Average (or percentile) pairwise distance
    
    Example:
        >>> objectives = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        >>> dist = average_pairwise_distance(objectives)
        >>> print(f"Avg pairwise distance: {dist:.4f}")
    """
    objectives = np.atleast_2d(objectives)
    
    if len(objectives) < 2:
        return 0.0
    
    # Compute pairwise distances
    distances = pdist(objectives, metric=metric)
    
    if percentile is not None:
        return float(np.percentile(distances, percentile))
    else:
        return float(np.mean(distances))


def spacing(objectives: np.ndarray) -> float:
    """
    Compute spacing metric (distribution uniformity).
    
    Spacing measures how evenly distributed solutions are along the Pareto front.
    Lower spacing = more uniform distribution.
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
    
    Returns:
        s: Spacing value (lower is better)
    
    Example:
        >>> objectives = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        >>> s = spacing(objectives)
        >>> print(f"Spacing: {s:.4f}")
    """
    objectives = np.atleast_2d(objectives)
    
    if len(objectives) < 2:
        return 0.0
    
    # For each point, find distance to nearest neighbor
    min_distances = []
    
    for i in range(len(objectives)):
        # Distances to all other points
        distances = np.linalg.norm(objectives - objectives[i], axis=1)
        # Exclude self (distance = 0)
        distances = distances[distances > 0]
        
        if len(distances) > 0:
            min_distances.append(np.min(distances))
    
    if len(min_distances) == 0:
        return 0.0
    
    # Spacing is standard deviation of nearest neighbor distances
    mean_dist = np.mean(min_distances)
    spacing_value = np.sqrt(np.mean((min_distances - mean_dist) ** 2))
    
    return float(spacing_value)


def spread(objectives: np.ndarray,
        reference_points: Optional[np.ndarray] = None) -> float:
    """
    Compute spread (delta) metric.
    
    Spread measures the extent of solutions along the Pareto front,
    including both extremes and distribution.
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
        reference_points: Extreme points of true Pareto front (optional)
    
    Returns:
        delta: Spread value (lower is better for uniformity)
    
    Example:
        >>> objectives = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        >>> delta = spread(objectives)
        >>> print(f"Spread: {delta:.4f}")
    """
    objectives = np.atleast_2d(objectives)
    n = len(objectives)
    
    if n < 2:
        return 0.0
    
    # Compute consecutive distances
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_indices]
    
    consecutive_distances = []
    for i in range(n - 1):
        dist = np.linalg.norm(sorted_obj[i + 1] - sorted_obj[i])
        consecutive_distances.append(dist)
    
    d_mean = np.mean(consecutive_distances)
    
    # Distance to extremes
    if reference_points is not None:
        d_f = np.linalg.norm(sorted_obj[0] - reference_points[0])
        d_l = np.linalg.norm(sorted_obj[-1] - reference_points[1])
    else:
        # Use ideal extremes
        d_f = np.linalg.norm(sorted_obj[0])
        d_l = np.linalg.norm(sorted_obj[-1])
    
    # Spread formula
    numerator = d_f + d_l + np.sum(np.abs(consecutive_distances - d_mean))
    denominator = d_f + d_l + (n - 1) * d_mean
    
    if denominator == 0:
        return 0.0
    
    delta = numerator / denominator
    
    return float(delta)


def generational_distance(objectives: np.ndarray,
                        true_pareto_front: np.ndarray,
                        p: float = 2.0) -> float:
    """
    Compute Generational Distance (GD).
    
    GD measures the average distance from obtained solutions to the true Pareto front.
    Lower GD = closer to true Pareto front.
    
    Args:
        objectives: Obtained objective values, shape (N, num_objectives)
        true_pareto_front: True Pareto front, shape (M, num_objectives)
        p: Distance power (default: 2 for Euclidean)
    
    Returns:
        gd: Generational distance
    
    Example:
        >>> objectives = np.array([[0.2, 0.8], [0.5, 0.5]])
        >>> true_pf = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        >>> gd = generational_distance(objectives, true_pf)
        >>> print(f"GD: {gd:.4f}")
    """
    objectives = np.atleast_2d(objectives)
    true_pareto_front = np.atleast_2d(true_pareto_front)
    
    # For each obtained point, find closest point on true Pareto front
    distances = cdist(objectives, true_pareto_front, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    
    # GD = (mean of p-th powers)^(1/p)
    gd = np.power(np.mean(np.power(min_distances, p)), 1.0 / p)
    
    return float(gd)


def inverted_generational_distance(objectives: np.ndarray,
                                true_pareto_front: np.ndarray,
                                p: float = 2.0) -> float:
    """
    Compute Inverted Generational Distance (IGD).
    
    IGD measures the average distance from true Pareto front to obtained solutions.
    Lower IGD = better coverage and convergence.
    
    Args:
        objectives: Obtained objective values, shape (N, num_objectives)
        true_pareto_front: True Pareto front, shape (M, num_objectives)
        p: Distance power (default: 2 for Euclidean)
    
    Returns:
        igd: Inverted generational distance
    """
    objectives = np.atleast_2d(objectives)
    true_pareto_front = np.atleast_2d(true_pareto_front)
    
    # For each true Pareto point, find closest obtained point
    distances = cdist(true_pareto_front, objectives, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    
    # IGD = (mean of p-th powers)^(1/p)
    igd = np.power(np.mean(np.power(min_distances, p)), 1.0 / p)
    
    return float(igd)


def compute_all_traditional_metrics(objectives: np.ndarray,
                                reference_point: np.ndarray,
                                true_pareto_front: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all traditional metrics at once.
    
    Args:
        objectives: Objective values
        reference_point: Reference point for HV and R2
        true_pareto_front: True Pareto front (optional, for GD/IGD)
    
    Returns:
        metrics: Dictionary of all computed metrics
    """
    metrics = {}
    
    # Quality metrics
    metrics['hypervolume'] = hypervolume(objectives, reference_point)
    metrics['r2_indicator'] = r2_indicator(objectives, reference_point)
    
    # Diversity metrics
    metrics['avg_pairwise_distance'] = average_pairwise_distance(objectives)
    metrics['spacing'] = spacing(objectives)
    metrics['spread'] = spread(objectives)
    
    # Convergence metrics (if true PF available)
    if true_pareto_front is not None:
        metrics['generational_distance'] = generational_distance(
            objectives, true_pareto_front
        )
        metrics['inverted_generational_distance'] = inverted_generational_distance(
            objectives, true_pareto_front
        )
    
    return metrics


# Example usage and tests
if __name__ == '__main__':
    print("Testing Traditional Multi-Objective Metrics")
    print("=" * 60)
    
    # Test data: simple 2-objective case
    objectives = np.array([
        [0.1, 0.9],
        [0.2, 0.7],
        [0.4, 0.5],
        [0.6, 0.4],
        [0.8, 0.2],
        [0.9, 0.1]
    ])
    reference_point = np.array([1.0, 1.0])
    
    print("\nTest objectives:")
    print(objectives)
    
    print("\n" + "=" * 60)
    print("Computing metrics...")
    print("=" * 60)
    
    # Hypervolume
    hv = hypervolume(objectives, reference_point)
    print(f"\nHypervolume: {hv:.4f}")
    
    # R2
    r2 = r2_indicator(objectives, reference_point)
    print(f"R2 Indicator: {r2:.4f}")
    
    # Diversity metrics
    avg_dist = average_pairwise_distance(objectives)
    print(f"Avg Pairwise Distance: {avg_dist:.4f}")
    
    s = spacing(objectives)
    print(f"Spacing: {s:.4f}")
    
    delta = spread(objectives)
    print(f"Spread: {delta:.4f}")
    
    # All at once
    print("\n" + "=" * 60)
    print("All metrics:")
    print("=" * 60)
    all_metrics = compute_all_traditional_metrics(objectives, reference_point)
    for name, value in all_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")