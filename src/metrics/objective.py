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