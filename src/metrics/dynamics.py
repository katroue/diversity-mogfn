import numpy as np

def replay_buffer_diversity(replay_buffer: List[Trajectory], 
                            metric: str = 'trajectory_distance',
                            sample_size: int = None) -> float:
    """
    Compute Replay Buffer Diversity (RBD).
    
    RBD measures how diverse the experience stored in the replay buffer is.
    Higher diversity in training data should lead to more diverse outputs.
    
    Args:
        replay_buffer: List of trajectories stored in replay buffer
        metric: Diversity metric to use:
                'trajectory_distance' - average pairwise trajectory distance
                'objective_diversity' - diversity in objective space
                'state_coverage' - unique states visited
        sample_size: Number of trajectories to sample (None = use all)
    
    Returns:
        rbd: Replay buffer diversity score
    """
    if len(replay_buffer) == 0:
        return 0.0
    
    # Sample from buffer if specified
    if sample_size is not None and sample_size < len(replay_buffer):
        indices = np.random.choice(len(replay_buffer), sample_size, replace=False)
        buffer_sample = [replay_buffer[i] for i in indices]
    else:
        buffer_sample = replay_buffer
    
    if metric == 'trajectory_distance':
        # Compute average pairwise trajectory distance using edit distance
        if len(buffer_sample) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(buffer_sample)):
            for j in range(i + 1, len(buffer_sample)):
                # Edit distance on action sequences
                actions_i = buffer_sample[i].actions
                actions_j = buffer_sample[j].actions
                
                # Levenshtein distance
                m, n = len(actions_i), len(actions_j)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for ii in range(m + 1):
                    dp[ii][0] = ii
                for jj in range(n + 1):
                    dp[0][jj] = jj
                
                for ii in range(1, m + 1):
                    for jj in range(1, n + 1):
                        if actions_i[ii-1] == actions_j[jj-1]:
                            dp[ii][jj] = dp[ii-1][jj-1]
                        else:
                            dp[ii][jj] = 1 + min(dp[ii-1][jj], dp[ii][jj-1], dp[ii-1][jj-1])
                
                edit_dist = dp[m][n]
                normalized_dist = edit_dist / max(m, n) if max(m, n) > 0 else 0
                total_distance += normalized_dist
                count += 1
        
        rbd = total_distance / count if count > 0 else 0.0
    
    elif metric == 'objective_diversity':
        # Diversity in objective space (rewards)
        from scipy.spatial.distance import pdist
        
        objectives = []
        for traj in buffer_sample:
            if isinstance(traj.reward, torch.Tensor):
                obj = traj.reward.detach().cpu().numpy()
            else:
                obj = np.array(traj.reward) if isinstance(traj.reward, (list, tuple)) else np.array([traj.reward])
            objectives.append(obj)
        
        objectives = np.array(objectives)
        
        if len(objectives) < 2 or objectives.shape[1] == 0:
            return 0.0
        
        # Average pairwise distance in objective space
        distances = pdist(objectives, metric='euclidean')
        rbd = float(np.mean(distances))
    
    elif metric == 'state_coverage':
        # Count unique terminal states visited
        unique_states = set()
        
        for traj in buffer_sample:
            # Get terminal state
            terminal_state = traj.states[-1]
            
            # Convert to hashable tuple
            if isinstance(terminal_state, torch.Tensor):
                state_tuple = tuple(terminal_state.detach().cpu().numpy().flatten())
            else:
                state_tuple = tuple(np.array(terminal_state).flatten())
            
            unique_states.add(state_tuple)
        
        # Diversity = proportion of unique states
        rbd = len(unique_states) / len(buffer_sample) if len(buffer_sample) > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return float(rbd)