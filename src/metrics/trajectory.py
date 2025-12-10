import numpy as np

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
            max_possible += max(len(trajectories[i].actions), len(trajectories[j].actions))

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