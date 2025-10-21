import numpy as np
from typing import List, Any
try:
    import torch
except Exception:
    torch = None

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
def multi_path_diversity(trajectories: List[Any],
                        method: str = 'entropy',
                        similarity_threshold: float = 0.9) -> float:
    """
    Compute Multi-Path Diversity (MPD).
    
    MPD measures how many different paths lead to the same terminal state.
    Higher MPD indicates the GFlowNet exploits multiple paths to solutions.
    
    MPD = E_x[log(N_paths(x) + 1)]
    
    Args:
        trajectories: List of trajectories sampled from GFlowNet
        method: 'entropy' for log-based or 'count' for average count
        similarity_threshold: Threshold for considering trajectories as "same path"
                            (only used for approximate counting)
    
    Returns:
        mpd: Multi-path diversity score
    
    Example:
        >>> trajectories = sample_trajectories(gflownet, num_samples=100)
        >>> mpd = multi_path_diversity(trajectories)
        >>> print(f"MPD: {mpd:.4f}")
    """
    if len(trajectories) == 0:
        return 0.0
    
    # Group trajectories by terminal state
    terminal_state_groups = {}
    
    for traj in trajectories:
        # Get terminal state
        terminal_state = traj.states[-1]
        
        # Convert to hashable key
        if torch is not None and isinstance(terminal_state, torch.Tensor):
            state_key = tuple(terminal_state.detach().cpu().numpy().flatten().round(6))
        else:
            state_key = tuple(np.array(terminal_state).flatten().round(6))
        
        if state_key not in terminal_state_groups:
            terminal_state_groups[state_key] = []
        
        terminal_state_groups[state_key].append(traj)
    
    # For each terminal state, count distinct paths
    log_path_counts = []
    
    for state_key, trajs in terminal_state_groups.items():
        if len(trajs) == 1:
            # Only one trajectory to this state
            num_unique_paths = 1
        else:
            # Estimate number of unique paths using trajectory similarity
            num_unique_paths = _count_unique_trajectories(trajs, similarity_threshold)
        
        if method == 'entropy':
            # Log scale (as in paper definition)
            log_count = np.log(num_unique_paths + 1)
            log_path_counts.append(log_count)
        elif method == 'count':
            # Direct count
            log_path_counts.append(num_unique_paths)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Average across all terminal states
    mpd = float(np.mean(log_path_counts)) if log_path_counts else 0.0
    
    return mpd


def _count_unique_trajectories(trajectories: List[Any], 
                            threshold: float = 0.9) -> int:
    """
    Estimate number of unique trajectories using similarity clustering.
    
    Two trajectories are considered different if their similarity < threshold.
    
    Args:
        trajectories: List of trajectories to the same terminal state
        threshold: Similarity threshold (0-1)
    
    Returns:
        count: Estimated number of unique paths
    """
    if len(trajectories) <= 1:
        return len(trajectories)
    
    n = len(trajectories)
    
    # Compute pairwise trajectory similarities using normalized edit distance
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        similarity_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = _trajectory_similarity(trajectories[i], trajectories[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    # Greedy clustering: count trajectories as unique if no similar one exists
    unique_count = 0
    marked = set()
    
    for i in range(n):
        if i in marked:
            continue
        
        # This is a unique path
        unique_count += 1
        
        # Mark all similar trajectories
        for j in range(n):
            if similarity_matrix[i, j] >= threshold:
                marked.add(j)
    
    return unique_count
        
        # Mark all similar trajectories
def _trajectory_similarity(traj1: Any, traj2: Any) -> float:
    """
    Compute similarity between two trajectories using normalized edit distance.
    
    Returns similarity in [0, 1], where 1 = identical trajectories.
    """
    actions1 = traj1.actions
    actions2 = traj2.actions
    
    # Compute edit distance
    m, n = len(actions1), len(actions2)
    
    if m == 0 and n == 0:
        return 1.0
    
    if m == 0 or n == 0:
        return 0.0
    
    # Dynamic programming for edit distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if actions1[i-1] == actions2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    edit_distance = dp[m][n]
    max_length = max(m, n)
    
    # Convert to similarity (1 - normalized distance)
    similarity = 1.0 - (edit_distance / max_length)
    
    return similarity
    edit_distance = dp[m][n]
    max_length = max(m, n)
    
    # Convert to similarity (1 - normalized distance)
    similarity = 1.0 - (edit_distance / max_length)
    
    return similarity