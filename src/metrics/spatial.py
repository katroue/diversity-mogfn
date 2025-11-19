import numpy as np

def mode_coverage_entropy(solutions_obj_space, eps='auto',min_samples=5):
    """
    Compute MCE using DBSCAN clustering in objective space

    Args:
        solutions_obj_space (_type_): Array of shape (N,m) where m = num objectives
        eps: DBSCAN radius parameter (auto-tuned if 'auto')
        min_samples: DBSCAN minimum samples per cluster
    
    Returns:
        mce: FLoat in [0,1], higher = better mode coverage
        num_modes: Integer, number of discovered modes
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    
    # FIXED: Convert PyTorch tensor to numpy if needed
    if hasattr(solutions_obj_space, 'cpu'):
        solutions_obj_space = solutions_obj_space.cpu().numpy()
    
    N = len(solutions_obj_space)
    
    if N < min_samples:
        return 0.0, 0
    
    # Adjust min_samples if dataset is too small
    effective_min_samples = min(min_samples, max(2, N // 5))
    
    # Auto-tune eps using k-distance graph
    if eps == 'auto':
        try:
            neigh = NearestNeighbors(n_neighbors=effective_min_samples)
            nbrs = neigh.fit(solutions_obj_space)
            distances, _ = nbrs.kneighbors(solutions_obj_space)
            k_distances = np.sort(distances[:, -1])

            # FIXED: Improved elbow detection to avoid outliers
            # Instead of using argmax on full array, use percentile-based approach
            # to find elbow in the main body of the distribution (ignore tail outliers)
            n_consider = int(len(k_distances) * 0.90)  # Consider first 90%

            if n_consider < 2:
                n_consider = len(k_distances)

            diffs = np.diff(k_distances[:n_consider])

            if len(diffs) > 0:
                elbow_idx = np.argmax(diffs)
                eps_candidate = k_distances[elbow_idx]
            else:
                eps_candidate = 0

            # FIXED: Validate eps is reasonable (not too large)
            # eps should not exceed 75th percentile of k-distances
            max_reasonable_eps = np.percentile(k_distances, 75)

            if eps_candidate <= 0 or np.isnan(eps_candidate) or eps_candidate > max_reasonable_eps:
                # Fallback: use mean of ALL k-distances (including zeros)
                # This gives a more conservative eps that accounts for duplicates
                eps = np.mean(k_distances)
                if eps <= 0 or np.isnan(eps):
                    # All distances are zero (all duplicates) - use small value
                    eps = np.std(solutions_obj_space) / 10
                    if eps <= 0 or np.isnan(eps):
                        eps = 0.05  # Conservative default
            else:
                eps = float(eps_candidate)
        except Exception as e:
            # If auto-tuning fails, use a reasonable default
            eps = np.std(solutions_obj_space) / 2
            if eps <= 0 or np.isnan(eps):
                eps = 0.1
    
    # Ensure eps is valid
    eps = max(float(eps), 1e-6)  # Minimum threshold
    
    # Cluster with DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=effective_min_samples)
    labels = clustering.fit_predict(solutions_obj_space)
    
    # Get cluster distribution (excluding noise label -1)
    unique_labels = set(labels) - {-1}
    num_modes = len(unique_labels)
    
    if num_modes <= 1:
        return 0.0, num_modes
    
    # Compute entropy
    cluster_sizes = []
    
    for label in unique_labels:
        cluster_sizes.append(np.sum(labels == label))
    probs = np.array(cluster_sizes) / N
    entropy = -np.sum(probs * np.log(probs + 1e-10))  # Add small value to avoid log(0)
    
    # Normalize by maximum entropy (uniform distribution)
    max_entropy = np.log2(num_modes)
    mce = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return mce, num_modes


def pairwise_minimum_distance(objectives: np.ndarray, top_k: int = None) -> float:
    """
    Compute Pairwise Minimum Distance (PMD) for top-K solutions.
    
    PMD = min_{i≠j} d(x_i, x_j)
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
        top_k: Number of top solutions to consider (None = all)
    
    Returns:
        pmd: Minimum pairwise distance
    """
    from scipy.spatial.distance import pdist
    
    # FIXED: Convert PyTorch tensor to numpy if needed
    if hasattr(objectives, 'cpu'):
        objectives = objectives.cpu().numpy()
    
    # Select top-K non-dominated solutions if specified
    if top_k is not None and top_k < len(objectives):
        # Find non-dominated solutions
        n = len(objectives)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                # FIXED: objectives is now guaranteed to be numpy array
                if i != j and np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_dominated[i] = True
                    break
        
        non_dominated = objectives[~is_dominated]
        
        if len(non_dominated) >= top_k:
            objectives = non_dominated[:top_k]
        else:
            # Take all non-dominated + best dominated
            dominated = objectives[is_dominated]
            sums = np.sum(dominated, axis=1)
            best_dominated = dominated[np.argsort(sums)[:top_k - len(non_dominated)]]
            objectives = np.vstack([non_dominated, best_dominated])
    
    # Compute minimum pairwise distance
    if len(objectives) < 2:
        return 0.0
    
    distances = pdist(objectives, metric='euclidean')

    return float(np.min(distances))


def num_unique_solutions(objectives: np.ndarray, tolerance: float = 1e-9) -> int:
    """
    Count the number of unique solutions in objective space.

    This metric is useful for detecting solution duplication and diversity collapse.
    Higher values indicate more diverse solution sets.

    Args:
        objectives: Objective values, shape (N, num_objectives)
        tolerance: Tolerance for considering two solutions as identical
                (default: 1e-9 for near-exact matching)

    Returns:
        num_unique: Number of unique solutions

    Example:
        >>> objectives = np.array([[0.1, 0.9], [0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        >>> n_unique = num_unique_solutions(objectives)
        >>> print(f"Unique solutions: {n_unique}")  # Should print 3
    """
    # Convert PyTorch tensor to numpy if needed
    if hasattr(objectives, 'cpu'):
        objectives = objectives.cpu().numpy()

    objectives = np.atleast_2d(objectives)

    if len(objectives) == 0:
        return 0

    if len(objectives) == 1:
        return 1

    # Use numpy's unique with tolerance for floating point comparisons
    # Round to handle floating point precision issues
    if tolerance > 0:
        # Determine number of decimal places from tolerance
        decimals = max(0, int(-np.log10(tolerance)))
        objectives_rounded = np.round(objectives, decimals=decimals)
        unique_objectives = np.unique(objectives_rounded, axis=0)
    else:
        unique_objectives = np.unique(objectives, axis=0)

    return int(len(unique_objectives))