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
            
            # Heuristic: choose eps at the elbow point
            diffs = np.diff(k_distances)
            eps_candidate = k_distances[np.argmax(diffs)] if len(diffs) > 0 else 0
            
            # FIXED: Ensure eps is always positive and reasonable
            if eps_candidate <= 0 or np.isnan(eps_candidate):
                # Fallback: use median of k-distances
                eps = np.median(k_distances)
                if eps <= 0 or np.isnan(eps):
                    # Last resort: use mean distance
                    eps = np.mean(distances[:, -1])
                    if eps <= 0 or np.isnan(eps):
                        # Ultimate fallback: use standard deviation
                        eps = np.std(solutions_obj_space) / 2
                        if eps <= 0:
                            eps = 0.1
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