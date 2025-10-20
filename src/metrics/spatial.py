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
    import numpy as np
    
    N = len(solutions_obj_space)
    
    if N < min_samples:
        return 0.0, 0
    
    # Auto-tune eps using k-distance graph\
    if eps == 'auto':
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(solutions_obj_space)
        distances, _ = nbrs.kneighbors(solutions_obj_space)
        k_distances = np.sort(distances[:, -1])
        # Heuristic: choose eps at the elbow point
        diffs = np.diff(k_distances)
        eps = k_distances[np.argmax(diffs)]
    
    #Cluster with DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
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