from typing import Union, Dict, List
import numpy as np

def flow_concentration_index(state_visit_counts: Union[Dict, np.ndarray, List],
                            method: str = 'gini') -> float:
    """
    Compute Flow Concentration Index (FCI).
    
    FCI measures how concentrated the learned flow is across states.
    Uses Gini coefficient of state visitation frequencies.
    
    FCI = 0: Flow uniformly distributed (ideal diversity)
    FCI = 1: Flow concentrated in few states (poor diversity)
    
    Args:
        state_visit_counts: State visitation counts, can be:
                        - Dict mapping state -> count
                        - Array of visit counts
                        - List of visit counts
        method: 'gini' for Gini coefficient or 'entropy' for normalized entropy
    
    Returns:
        fci: Flow concentration index in [0, 1]
    
    Example:
        >>> # Uniform flow (good diversity)
        >>> counts = [10, 10, 10, 10, 10]
        >>> fci = flow_concentration_index(counts)
        >>> print(f"FCI: {fci:.4f}")  # Should be close to 0
        
        >>> # Concentrated flow (poor diversity)
        >>> counts = [100, 1, 1, 1, 1]
        >>> fci = flow_concentration_index(counts)
        >>> print(f"FCI: {fci:.4f}")  # Should be close to 1
    """
    # Convert input to numpy array
    if isinstance(state_visit_counts, dict):
        counts = np.array(list(state_visit_counts.values()), dtype=float)
    else:
        counts = np.array(state_visit_counts, dtype=float)
    
    # Remove zeros
    counts = counts[counts > 0]
    
    if len(counts) == 0:
        return 0.0
    
    if len(counts) == 1:
        return 1.0  # All flow in one state = maximum concentration
    
    if method == 'gini':
        # Compute Gini coefficient
        # Sort counts
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        
        # Compute cumulative sum
        cumsum = np.cumsum(sorted_counts)
        
        # Gini coefficient formula
        # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
        
        fci = float(gini)
    
    elif method == 'entropy':
        # Normalized entropy (inverse as concentration measure)
        # Higher entropy = more uniform = lower concentration
        
        # Compute probabilities
        probs = counts / np.sum(counts)
        
        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by maximum entropy (uniform distribution)
        max_entropy = np.log2(len(counts))
        
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
            # Convert to concentration (inverse of uniformity)
            fci = 1.0 - normalized_entropy
        else:
            fci = 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gini' or 'entropy'")
    
    return float(fci)