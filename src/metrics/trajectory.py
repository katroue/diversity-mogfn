def trajectory_diversity_score(trajectories):
    """
    Compute TDS using edit distance on action sequences.

    Args:
        trajectories (_type_): _description_

    Returns:
        _type_: _description_
    """
    N = len(trajectories)
    if N < 2:
        return 0.0
    
    total_distance = 0.0
    max_possible = 0 # Maximum possible distance for normalization
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = edit_distance(trajectories[i].actions, trajectories[j].actions)
            total_distance += dist
            max_possible += max(len(trajectories[i].actions), len(trajectories[j].actions)
                            )
    
    if max_possible == 0:
        return 0.0
    
    tds = total_distance / max_possible
    return tds

def edit_distance(seq1, seq2):
    """Levenshtein distance between two sequences."""
    
    len_seq1 = len(seq1) + 1
    len_seq2 = len(seq2) + 1

    # Create a distance matrix
    matrix = np.zeros((len_seq1, len_seq2), dtype=int)

    for i in range(len_seq1):
        matrix[i][0] = i
    for j in range(len_seq2):
        matrix[0][j] = j

    for i in range(1, len_seq1):
        for j in range(1, len_seq2):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(matrix[i-1][j] + 1,      # Deletion
                            matrix[i][j-1] + 1,      # Insertion
                            matrix[i-1][j-1] + cost) # Substitution

    return matrix[-1][-1]


def multi_path_diversity(gflownet, terminal_states, samples_per_states=100):
    """
    Estimate MPD by sampling multiple paths to each terminal state.
    
    Args: 
        gflownet (_type_): GFlowNet model with backward policy
        terminal_states (_type_): List of terminal states to sample paths for
        samples_per_states (int, optional): Number of trajectories to sample per terminal state
        
        Returns:
        mpd: Average trajectory diversity score across terminal states
        path_counts: Dict mapping state -> estimated path count
    """
    
    import numpy as np
    from collections import defaultdict
    
    log_path_diversities = []
    path_counts = {}
    
    for terminal_state in terminal_states:
        trajectories = []
        
        for _ in range(samples_per_states):
            traj = gflownet.sample_backward_trajectory(final_state=terminal_state)
            trajectories.append(traj)
        
        unique_paths - estimate_unique_paths(trajectories)
        
        log_diversity = np.log(unique_paths + 1)
        log_path_diversities
        path_counts[terminal_state] = unique_paths
    
    mpd = np.mean(log_path_diversities)
    return mpd, path_counts

def estimate_unique_paths(trajectories, treshold=0.9):
    """Estimate number of unique paths using similarirty treshold.
    
    Two trajectories are considered the same path if their similarirty < threshold."""
    
    from scipy.spatial.distance import pdist, squareform
    import numpy as np
    
    if len(trajectories) <= 1:
        return len(trajectories)
    
    
    n = len
    traj_matrix = np.zeros((n, max(len(t.actions) for t in trajectories)), dtype=int)
    
    for i, traj in enumerate(trajectories):
        traj_matrix[i, :len(traj.actions)] = traj.actions
    pairwise_dists = squareform(pdist(traj_matrix, metric='hamming')) 
    
    visited = set()
    unique_count = 0
    
    for i in range(n):
        if i in visited:
            continue
        unique_count += 1
        for j in range(i + 1, n):
            if pairwise_dists[i, j] < threshold:
                visited.add(j)
    
    return unique_count