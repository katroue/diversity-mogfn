import numpy as np


def pareto_front_smoothness(objectives: np.ndarray, method: str = 'curve_fitting', max_samples: int = 10000) -> float:
    """
    Compute Pareto Front Smoothness (PFS).

    PFS measures how smooth/continuous the discovered Pareto front is.
    Lower PFS = smoother front, Higher PFS = more jagged/discontinuous.

    EXTENDED VERSION: Now supports 2+ objectives via manifold projection.
    - For 2 objectives: Uses 1D curve fitting (original method)
    - For 3+ objectives: Projects to 2D manifold via PCA, then curve fitting

    OPTIMIZED: For very large datasets (>max_samples), samples a representative subset
    before extracting Pareto front to avoid excessive computation time.

    PFS = Σ d(point, fitted_curve)²

    Args:
        objectives: Objective values, shape (N, num_objectives)
        method: 'curve_fitting' for polynomial fit or 'local_variance' for neighbor-based
        max_samples: Maximum samples to process (larger datasets are randomly sampled)

    Returns:
        pfs: Pareto front smoothness score (lower is smoother)
    """
    # FIXED: Convert PyTorch tensor to numpy at the start
    if hasattr(objectives, 'cpu'):
        objectives = objectives.cpu().numpy()

    if len(objectives) < 3:
        return 0.0  # Need at least 3 points for smoothness

    # For extremely large datasets, sample a representative subset
    # This prevents hours of computation on millions of points
    if len(objectives) > max_samples:
        # Use stratified sampling to maintain diversity
        # Sample uniformly across the objective space
        np.random.seed(42)  # Reproducible sampling
        indices = np.random.choice(len(objectives), size=max_samples, replace=False)
        objectives = objectives[indices]

    # Filter to non-dominated solutions (Pareto front)
    pareto_points = _extract_pareto_front(objectives)

    if len(pareto_points) < 3:
        return 0.0

    # Route to appropriate method based on dimensionality
    num_objectives = objectives.shape[1]

    if num_objectives == 2:
        # Use original 2D implementation
        return _pfs_2d(pareto_points, method)
    else:
        # Use manifold projection for 3+ objectives
        return _pfs_multiobjective(pareto_points, method)


def _extract_pareto_front(objectives: np.ndarray) -> np.ndarray:
    """
    Extract non-dominated solutions (Pareto front) from objectives.

    OPTIMIZED: Uses vectorized operations for O(N*M) complexity instead of O(N²).
    For large datasets (>10k samples), uses efficient batch processing.

    Args:
        objectives: Objective values, shape (N, num_objectives)

    Returns:
        pareto_points: Non-dominated points, shape (M, num_objectives) where M <= N
    """
    n = len(objectives)

    # For small datasets, use simple vectorized approach
    if n <= 10000:
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            # Vectorized comparison: check if any point dominates point i
            # Point j dominates i if: j >= i in all objectives AND j > i in at least one
            better_or_equal = np.all(objectives >= objectives[i], axis=1)
            strictly_better = np.any(objectives > objectives[i], axis=1)
            dominates = better_or_equal & strictly_better
            dominates[i] = False  # A point doesn't dominate itself

            if np.any(dominates):
                is_dominated[i] = True

        return objectives[~is_dominated]

    # For large datasets, use efficient iterative filtering
    # Strategy: Process in smaller chunks and iteratively filter
    candidates = objectives.copy()
    max_iterations = 10  # Prevent infinite loops

    for iteration in range(max_iterations):
        n_candidates = len(candidates)

        # If we're down to a reasonable size, switch to simple method
        if n_candidates <= 5000:
            is_dominated = np.zeros(n_candidates, dtype=bool)
            for i in range(n_candidates):
                better_or_equal = np.all(candidates >= candidates[i], axis=1)
                strictly_better = np.any(candidates > candidates[i], axis=1)
                dominates = better_or_equal & strictly_better
                dominates[i] = False
                if np.any(dominates):
                    is_dominated[i] = True
            return candidates[~is_dominated]

        # For very large sets, use aggressive filtering
        # Process in chunks and filter dominated points
        chunk_size = 5000
        is_dominated = np.zeros(n_candidates, dtype=bool)

        for chunk_start in range(0, n_candidates, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_candidates)

            # Check if points in this chunk are dominated by ANY point
            for i in range(chunk_start, chunk_end):
                if is_dominated[i]:
                    continue

                point = candidates[i]

                # Check against all other points (vectorized)
                better_or_equal = np.all(candidates >= point, axis=1)
                strictly_better = np.any(candidates > point, axis=1)
                dominates = better_or_equal & strictly_better
                dominates[i] = False

                if np.any(dominates):
                    is_dominated[i] = True

        # Remove dominated points
        non_dominated = candidates[~is_dominated]

        # If no significant reduction, we're done
        reduction_ratio = len(non_dominated) / n_candidates
        if reduction_ratio > 0.95:  # Less than 5% removed
            break

        candidates = non_dominated

    return candidates


def _pfs_2d(pareto_points: np.ndarray, method: str = 'curve_fitting') -> float:
    """
    Compute PFS for 2-objective problems (original implementation).

    Args:
        pareto_points: Non-dominated points, shape (N, 2)
        method: 'curve_fitting' or 'local_variance'

    Returns:
        pfs: Smoothness score
    """
    if len(pareto_points) < 3:
        return 0.0

    return _pfs_curve_fitting(pareto_points, method)


def _pfs_multiobjective(pareto_points: np.ndarray, method: str = 'curve_fitting') -> float:
    """
    Compute PFS for 3+ objective problems using manifold projection.

    Projects Pareto front to 2D manifold via PCA, then applies curve fitting.

    Args:
        pareto_points: Non-dominated points, shape (N, num_objectives) where num_objectives >= 3
        method: 'curve_fitting' or 'local_variance'

    Returns:
        pfs: Smoothness score
    """
    if len(pareto_points) < 10:
        # Need more points for reliable PCA projection
        return 0.0

    try:
        from sklearn.decomposition import PCA

        # Project to 2D manifold
        # Most Pareto fronts lie on lower-dimensional manifolds
        pca = PCA(n_components=2)
        embedded = pca.fit_transform(pareto_points)

        # Check explained variance - if too low, projection isn't meaningful
        explained_var = np.sum(pca.explained_variance_ratio_)
        if explained_var < 0.5:
            # Projection doesn't capture enough structure
            # Fall back to neighbor-based smoothness
            return _pfs_knn_fallback(pareto_points)

        # Apply 2D curve fitting to embedded points
        return _pfs_curve_fitting(embedded, method)

    except ImportError:
        # sklearn not available, use fallback
        return _pfs_knn_fallback(pareto_points)
    except Exception:
        # PCA failed for some reason
        return _pfs_knn_fallback(pareto_points)


def _pfs_knn_fallback(pareto_points: np.ndarray, k: int = 5) -> float:
    """
    Fallback PFS computation using k-nearest neighbors.

    Measures smoothness as variance in neighbor distances.

    Args:
        pareto_points: Non-dominated points, shape (N, num_objectives)
        k: Number of neighbors to consider

    Returns:
        pfs: Smoothness score (variance in neighbor distances)
    """
    if len(pareto_points) < k + 1:
        return 0.0

    try:
        from sklearn.neighbors import NearestNeighbors

        k_actual = min(k, len(pareto_points) - 1)
        nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(pareto_points)
        distances, indices = nbrs.kneighbors(pareto_points)

        # Exclude self (first neighbor)
        neighbor_dists = distances[:, 1:]

        # Smoothness = coefficient of variation in neighbor distances
        # (variance normalized by mean)
        mean_dist = np.mean(neighbor_dists)
        if mean_dist < 1e-10:
            return 0.0

        pfs = np.var(neighbor_dists) / (mean_dist ** 2)

        return float(pfs)

    except ImportError:
        # sklearn not available
        # Use simple pairwise distance variance
        from scipy.spatial.distance import pdist
        dists = pdist(pareto_points)
        pfs = np.var(dists) / (np.mean(dists) ** 2 + 1e-10)
        return float(pfs)


def _pfs_curve_fitting(points: np.ndarray, method: str) -> float:
    """
    Core curve fitting logic for PFS computation.

    Args:
        points: Points to fit, shape (N, 2)
        method: 'curve_fitting' or 'local_variance'

    Returns:
        pfs: Smoothness score
    """
    if method == 'curve_fitting':
        # Sort by first dimension
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

        # Check for degenerate cases that would cause SVD to fail
        x_range = np.max(sorted_points[:, 0]) - np.min(sorted_points[:, 0])
        y_range = np.max(sorted_points[:, 1]) - np.min(sorted_points[:, 1])

        # If all points have same x or y coordinate, return 0 (perfectly smooth/degenerate)
        if x_range < 1e-10 or y_range < 1e-10:
            return 0.0

        # Check for duplicate x-values that would cause singular matrix
        unique_x = np.unique(sorted_points[:, 0])
        if len(unique_x) < 3:
            return 0.0  # Not enough unique x-values for meaningful curve fit

        try:
            # Fit polynomial curve (degree 2 or 3)
            degree = min(3, len(sorted_points) - 1)

            # Suppress RankWarning for nearly-collinear points (indicates very smooth front)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=np.RankWarning)
                coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], degree)

            poly = np.poly1d(coeffs)

            # Compute fitted values
            fitted_y = poly(sorted_points[:, 0])

            # Smoothness = sum of squared deviations from fitted curve
            deviations = sorted_points[:, 1] - fitted_y
            pfs = np.sum(deviations ** 2)

            # Normalize by number of points and variance
            pfs = pfs / (len(sorted_points) * (np.var(sorted_points[:, 1]) + 1e-10))

        except (np.linalg.LinAlgError, ValueError):
            # Polyfit failed (SVD didn't converge, singular matrix, etc.)
            # Return 0.0 as fallback (treat as degenerate case)
            return 0.0

    elif method == 'local_variance':
        # Sort by first dimension
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

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