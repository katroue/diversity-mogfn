import numpy as np

def preference_aligned_spread(gflownet, num_preferences=2, samples_per_pref=50):
    """Compute PAS by sampling multiple preferences and measuring spread
    

    Args:
        gflownet (_type_): Conditional GFLowNet (MOGFN-PC or HN-GFN)
        num_preferences (int, optional): Number of preference vectors to sample.
        samples_per_pref (int, optional): Solutions sampled per preference


    Returns:
    pas: Average spread across preferences
    spreads: List of spreads per preference (for analysis)
    """
    
    import numpy as np
    from spicy.spatial.distance import pdist
    
    spreads = []
    
    for _ in range(num_preferences):
        pref = sample_preference(gflownet.preference_dist)
        
        solutions_obj = []
        
        for _ in range(samples_per_pref):
            traj = gflownet.sample_trajectory(preference=pref)
            final_state = traj.final_state()
            obj_values = gflownet.env.get_objective(final_state)
            solutions_obj.append(obj_values)
        
        solutions_obj = np.array(solutions_obj)
        
        if len(solutions_obj) > 1:
            pairwise_dists = pdist(solutions_obj, metric='euclidean')
            spread = np.mean(pairwise_dists)
        else:
            spread = 0.0
        
        spreads.append(spread)
    
    pas = np.mean(spreads)
    return pas, spreads

def sample_preference(distribution_type='dirichlet', alpha=1.5, num_objectives=2):
    """Sample a preference vector from the specified distribution"""
    
    import numpy as np
    
    if distribution_type == 'dirichlet':
        pref = np.random.dirichlet(alpha * np.ones(num_objectives))
    elif distribution_type == 'uniform':
        pref = np.random.rand(num_objectives)
        pref /= np.sum(pref)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}") 
    
    return pref


def pareto_front_smoothness(objectives: np.ndarray, method: str = 'curve_fitting') -> float:
    """
    Compute Pareto Front Smoothness (PFS).
    
    PFS measures how smooth/continuous the discovered Pareto front is.
    Lower PFS = smoother front, Higher PFS = more jagged/discontinuous.
    
    PFS = Σ d(point, fitted_curve)²
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
        method: 'curve_fitting' for polynomial fit or 'local_variance' for neighbor-based
    
    Returns:
        pfs: Pareto front smoothness score (lower is smoother)
    """
    if len(objectives) < 3:
        return 0.0  # Need at least 3 points for smoothness
    
    # Filter to non-dominated solutions (Pareto front)
    n = len(objectives)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                is_dominated[i] = True
                break
    
    pareto_points = objectives[~is_dominated]
    
    if len(pareto_points) < 3:
        return 0.0
    
    if method == 'curve_fitting':
        # Sort by first objective
        sorted_indices = np.argsort(pareto_points[:, 0])
        sorted_points = pareto_points[sorted_indices]
        
        # Fit polynomial curve (degree 2 or 3)
        degree = min(3, len(sorted_points) - 1)
        coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], degree)
        poly = np.poly1d(coeffs)
        
        # Compute fitted values
        fitted_y = poly(sorted_points[:, 0])
        
        # Smoothness = sum of squared deviations from fitted curve
        deviations = sorted_points[:, 1] - fitted_y
        pfs = np.sum(deviations ** 2)
        
        # Normalize by number of points and variance
        pfs = pfs / (len(sorted_points) * (np.var(sorted_points[:, 1]) + 1e-10))
        
    elif method == 'local_variance':
        # Sort by first objective
        sorted_indices = np.argsort(pareto_points[:, 0])
        sorted_points = pareto_points[sorted_indices]
        
        # Compute local curvature using second derivatives
        if len(sorted_points) < 3:
            return 0.0
        
        total_curvature = 0.0
        for i in range(1, len(sorted_points) - 1):
            # Central difference approximation of second derivative
            dx1 = sorted_points[i, 0] - sorted_points[i-1, 0]
            dx2 = sorted_points[i+1, 0] - sorted_points[i, 0]
            dy1 = sorted_points[i, 1] - sorted_points[i-1, 1]
            dy2 = sorted_points[i+1, 1] - sorted_points[i, 1]
            
            if dx1 > 0 and dx2 > 0:
                slope1 = dy1 / dx1
                slope2 = dy2 / dx2
                curvature = abs(slope2 - slope1)
                total_curvature += curvature
        
        pfs = total_curvature / (len(sorted_points) - 2)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(pfs)