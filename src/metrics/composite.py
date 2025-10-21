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


def diversity_efficiency_ratio(objectives: np.ndarray,
                            training_time: float = None,
                            num_parameters: int = None,
                            diversity_metric: str = 'avg_distance',
                            normalize: bool = True) -> Dict[str, float]:
    """
    Compute Diversity-Efficiency Ratio (DER).
    
    DER measures diversity per unit of computational resources.
    Higher DER = better diversity for less computation (more efficient).
    
    DER = Diversity / (Training_time * Parameters)
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
        training_time: Training time in seconds (optional)
        num_parameters: Number of model parameters (optional)
        diversity_metric: Which diversity measure to use:
                        'avg_distance' - average pairwise distance
                        'spread' - average range across objectives
                        'entropy' - entropy-based diversity
        normalize: Whether to normalize by reasonable baselines
    
    Returns:
        results: Dictionary containing:
                - der: Diversity-efficiency ratio
                - diversity: Raw diversity value
                - training_time: Training time (if provided)
                - num_parameters: Number of parameters (if provided)
                - computational_cost: Combined cost metric
    
    Example:
        >>> objectives = np.array([[0.8, 0.3], [0.6, 0.5], [0.3, 0.8]])
        >>> results = diversity_efficiency_ratio(
        ...     objectives, 
        ...     training_time=3600,  # 1 hour
        ...     num_parameters=500000  # 500K params
        ... )
        >>> print(f"DER: {results['der']:.6f}")
    """
    from scipy.spatial.distance import pdist
    
    if len(objectives) < 2:
        return {
            'der': 0.0,
            'diversity': 0.0,
            'training_time': training_time,
            'num_parameters': num_parameters,
            'computational_cost': 0.0
        }
    
    # ===== 1. Compute Diversity =====
    if diversity_metric == 'avg_distance':
        # Average pairwise Euclidean distance
        distances = pdist(objectives, metric='euclidean')
        diversity = np.mean(distances)
        
    elif diversity_metric == 'spread':
        # Average range across objectives
        ranges = np.max(objectives, axis=0) - np.min(objectives, axis=0)
        diversity = np.mean(ranges)
        
    elif diversity_metric == 'entropy':
        # Cluster and compute entropy of distribution
        from sklearn.cluster import KMeans
        n_clusters = min(10, len(objectives))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(objectives)
        
        # Compute entropy of cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / len(objectives)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by max entropy
        max_entropy = np.log2(n_clusters)
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        
    else:
        raise ValueError(f"Unknown diversity_metric: {diversity_metric}")
    
    # ===== 2. Compute Computational Cost =====
    if training_time is None and num_parameters is None:
        # No cost information provided, return just diversity
        return {
            'der': float(diversity),  # Just return diversity
            'diversity': float(diversity),
            'training_time': None,
            'num_parameters': None,
            'computational_cost': 1.0  # Default to 1
        }
    
    computational_cost = 1.0
    
    if training_time is not None:
        if normalize:
            # Normalize by 1 hour (3600 seconds)
            time_factor = training_time / 3600.0
        else:
            time_factor = training_time
        computational_cost *= time_factor
    
    if num_parameters is not None:
        if normalize:
            # Normalize by 1 million parameters
            param_factor = num_parameters / 1e6
        else:
            param_factor = num_parameters
        computational_cost *= param_factor
    
    # Avoid division by zero
    if computational_cost == 0:
        computational_cost = 1e-10
    
    # ===== 3. Compute DER =====
    der = diversity / computational_cost
    
    return {
        'der': float(der),
        'diversity': float(diversity),
        'training_time': training_time,
        'num_parameters': num_parameters,
        'computational_cost': float(computational_cost),
        'diversity_metric': diversity_metric
    }


def compare_efficiency(results_list: List[Dict[str, Any]],
                    names: List[str] = None) -> Dict[str, List]:
    """
    Compare efficiency of multiple models/configurations.
    
    Args:
        results_list: List of dictionaries, each containing:
                    - objectives: np.ndarray
                    - training_time: float
                    - num_parameters: int
        names: Optional names for each configuration
    
    Returns:
        comparison: Dictionary with comparison statistics
    
    Example:
        >>> model_a = {
        ...     'objectives': objectives_a,
        ...     'training_time': 1800,
        ...     'num_parameters': 100000
        ... }
        >>> model_b = {
        ...     'objectives': objectives_b,
        ...     'training_time': 3600,
        ...     'num_parameters': 500000
        ... }
        >>> comparison = compare_efficiency([model_a, model_b], ['Small', 'Large'])
    """
    if names is None:
        names = [f"Model {i+1}" for i in range(len(results_list))]
    
    der_values = []
    diversity_values = []
    time_values = []
    param_values = []
    
    for result in results_list:
        der_result = diversity_efficiency_ratio(
            objectives=result['objectives'],
            training_time=result.get('training_time'),
            num_parameters=result.get('num_parameters')
        )
        
        der_values.append(der_result['der'])
        diversity_values.append(der_result['diversity'])
        time_values.append(result.get('training_time', 0))
        param_values.append(result.get('num_parameters', 0))
    
    # Find best by DER
    best_idx = np.argmax(der_values)
    
    return {
        'names': names,
        'der': der_values,
        'diversity': diversity_values,
        'training_time': time_values,
        'num_parameters': param_values,
        'best_model': names[best_idx],
        'best_der': der_values[best_idx]
    }


def plot_efficiency_comparison(comparison: Dict[str, List],
                            save_path: str = None):
    """
    Plot efficiency comparison across models.
    
    Creates a scatter plot: diversity vs computational cost, with DER as size.
    """
    import matplotlib.pyplot as plt
    
    names = comparison['names']
    diversity = np.array(comparison['diversity'])
    der = np.array(comparison['der'])
    
    # Compute computational cost
    time = np.array(comparison['training_time'])
    params = np.array(comparison['num_parameters'])
    cost = (time / 3600.0) * (params / 1e6)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot: x=cost, y=diversity, size=DER
    scatter = ax.scatter(cost, diversity, s=der*1000, alpha=0.6, c=range(len(names)), cmap='viridis')
    
    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name, (cost[i], diversity[i]), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=10)
    
    ax.set_xlabel('Computational Cost (Time × Parameters)', fontsize=12)
    ax.set_ylabel('Diversity', fontsize=12)
    ax.set_title('Diversity-Efficiency Comparison\n(Bubble size = DER)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved efficiency comparison to {save_path}")
    else:
        plt.show()
    
    return fig


# Example usage and tests
if __name__ == '__main__':
    print("Testing Diversity-Efficiency Ratio (DER)")
    print("=" * 60)
    
    # Test data
    objectives_small = np.array([
        [0.1, 0.9],
        [0.3, 0.7],
        [0.5, 0.5],
        [0.7, 0.3],
        [0.9, 0.1]
    ])
    
    objectives_large = np.array([
        [0.05, 0.95],
        [0.15, 0.85],
        [0.25, 0.75],
        [0.35, 0.65],
        [0.45, 0.55],
        [0.55, 0.45],
        [0.65, 0.35],
        [0.75, 0.25],
        [0.85, 0.15],
        [0.95, 0.05]
    ])
    
    print("\nTest 1: Small model (fast, few params)")
    der_small = diversity_efficiency_ratio(
        objectives_small,
        training_time=1800,    # 30 minutes
        num_parameters=100000   # 100K params
    )
    print(f"  Diversity: {der_small['diversity']:.4f}")
    print(f"  Training time: {der_small['training_time']/60:.1f} min")
    print(f"  Parameters: {der_small['num_parameters']/1000:.0f}K")
    print(f"  Computational cost: {der_small['computational_cost']:.4f}")
    print(f"  DER: {der_small['der']:.6f}")
    
    print("\nTest 2: Large model (slow, many params)")
    der_large = diversity_efficiency_ratio(
        objectives_large,
        training_time=7200,     # 2 hours
        num_parameters=1000000  # 1M params
    )
    print(f"  Diversity: {der_large['diversity']:.4f}")
    print(f"  Training time: {der_large['training_time']/60:.1f} min")
    print(f"  Parameters: {der_large['num_parameters']/1000:.0f}K")
    print(f"  Computational cost: {der_large['computational_cost']:.4f}")
    print(f"  DER: {der_large['der']:.6f}")
    
    print("\nTest 3: Efficiency comparison")
    if der_small['der'] > der_large['der']:
        print(f"  → Small model is more efficient! ({der_small['der']/der_large['der']:.2f}x better DER)")
    else:
        print(f"  → Large model is more efficient! ({der_large['der']/der_small['der']:.2f}x better DER)")
    
    print("\nTest 4: Compare multiple models")
    results = [
        {
            'objectives': objectives_small,
            'training_time': 1800,
            'num_parameters': 100000
        },
        {
            'objectives': objectives_large,
            'training_time': 7200,
            'num_parameters': 1000000
        },
        {
            'objectives': objectives_large * 0.9,  # Slightly worse diversity
            'training_time': 3600,
            'num_parameters': 500000
        }
    ]
    
    comparison = compare_efficiency(results, names=['Small', 'Large', 'Medium'])
    print(f"  Best model by DER: {comparison['best_model']}")
    print(f"  Best DER: {comparison['best_der']:.6f}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")