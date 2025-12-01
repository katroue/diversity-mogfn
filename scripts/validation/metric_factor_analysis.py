#!/usr/bin/env python3
"""
Perform factor analysis on diversity metrics.

This script implements Validation 1.3: Factor Analysis from
Phase 4: Metric Validation.

It uses PCA and Factor Analysis to identify key metric dimensions and
determine how many independent factors explain metric variance.

Usage:
    # Analyze ablation studies
    python scripts/validation/metric_factor_analysis.py --dataset ablations

    # Analyze factorial experiments
    python scripts/validation/metric_factor_analysis.py --dataset factorials

    # Analyze both
    python scripts/validation/metric_factor_analysis.py --dataset all
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Metric categories
METRIC_CATEGORIES = {
    'traditional': ['hypervolume', 'spacing'],
    'trajectory': ['tds', 'mpd'],
    'spatial': ['mce', 'num_unique_solutions'],
    'objective': ['pfs', 'pas'],
    'composite': ['qds']
}

ALL_METRICS = [m for metrics in METRIC_CATEGORIES.values() for m in metrics]


def load_data(dataset: str) -> pd.DataFrame:
    """Load experiment data based on dataset type."""
    if dataset == 'ablations':
        from compute_metric_correlations import load_ablation_data
        return load_ablation_data()
    elif dataset == 'factorials':
        from compute_metric_correlations import load_factorial_data
        return load_factorial_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def prepare_data_for_pca(df: pd.DataFrame) -> tuple:
    """
    Prepare data for PCA by selecting available metrics and handling missing values.

    Args:
        df: DataFrame with experiment results

    Returns:
        Tuple of (data_matrix, metric_names)
    """
    # Select metrics that are available
    available_metrics = []
    for metric in ALL_METRICS:
        if metric in df.columns:
            non_null = df[metric].notna().sum()
            if non_null > 10:  # Require at least 10 data points
                available_metrics.append(metric)

    print(f"\n✓ Selected {len(available_metrics)} metrics with sufficient data")

    # Extract data matrix
    X = df[available_metrics].copy()

    # Handle missing values (drop rows with any NaN)
    X_clean = X.dropna()
    print(f"✓ Data matrix: {X_clean.shape[0]} experiments × {X_clean.shape[1]} metrics")

    if X_clean.shape[0] < 10:
        raise ValueError(f"Insufficient data after removing NaNs: {X_clean.shape[0]} rows")

    return X_clean.values, available_metrics


def perform_pca(X: np.ndarray, metric_names: List[str]) -> Dict:
    """
    Perform Principal Component Analysis.

    Args:
        X: Data matrix (n_samples × n_features)
        metric_names: List of metric names

    Returns:
        Dictionary with PCA results
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n{'='*80}")
    print("PRINCIPAL COMPONENT ANALYSIS")
    print(f"{'='*80}")

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"\nExplained Variance by Component:")
    for i, (var, cum_var) in enumerate(zip(explained_var[:10], cumulative_var[:10]), 1):
        print(f"  PC{i:2d}: {var*100:5.2f}%  (cumulative: {cum_var*100:5.2f}%)")

    # Find number of components for 90% variance
    n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
    print(f"\n✓ {n_components_90} components explain 90% of variance")

    return {
        'pca': pca,
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'n_components_90': n_components_90,
        'loadings': pca.components_
    }


def plot_scree_plot(explained_var: np.ndarray, cumulative_var: np.ndarray,
                output_path: Path, title: str = "Scree Plot"):
    """
    Create scree plot showing explained variance.

    Args:
        explained_var: Explained variance ratio for each component
        cumulative_var: Cumulative explained variance
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual variance
    components = np.arange(1, len(explained_var) + 1)
    ax.bar(components, explained_var * 100, alpha=0.6, label='Individual Variance')

    # Plot cumulative variance
    ax.plot(components, cumulative_var * 100, 'ro-', linewidth=2,
        markersize=6, label='Cumulative Variance')

    # Add 90% threshold line
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2,
            label='90% Threshold', alpha=0.7)

    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0.5, min(20.5, len(components) + 0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved scree plot: {output_path}")

    plt.close()


def plot_loadings_heatmap(loadings: np.ndarray, metric_names: List[str],
                        n_components: int, output_path: Path,
                        title: str = "PCA Loadings"):
    """
    Create heatmap of PCA loadings.

    Args:
        loadings: PCA component loadings
        metric_names: List of metric names
        n_components: Number of components to display
        output_path: Path to save the plot
        title: Plot title
    """
    # Take first n_components
    loadings_subset = loadings[:n_components, :]

    # Create DataFrame for easier plotting
    loadings_df = pd.DataFrame(
        loadings_subset.T,
        index=metric_names,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    fig, ax = plt.subplots(figsize=(10, max(8, len(metric_names) * 0.3)))

    sns.heatmap(
        loadings_df,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Loading'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved loadings heatmap: {output_path}")

    plt.close()


def interpret_components(loadings: np.ndarray, metric_names: List[str],
                        n_components: int) -> pd.DataFrame:
    """
    Interpret principal components by identifying top loading metrics.

    Args:
        loadings: PCA component loadings
        metric_names: List of metric names
        n_components: Number of components to interpret

    Returns:
        DataFrame with component interpretations
    """
    interpretations = []

    print(f"\n{'='*80}")
    print("COMPONENT INTERPRETATION")
    print(f"{'='*80}")

    for i in range(min(n_components, loadings.shape[0])):
        component_loadings = loadings[i, :]

        # Get top positive and negative loadings
        pos_idx = np.argsort(component_loadings)[-3:][::-1]  # Top 3 positive
        neg_idx = np.argsort(component_loadings)[:3]          # Top 3 negative

        print(f"\nPC{i+1} (explains {loadings[i].var() * 100:.1f}% of metric variance):")

        print("  Strong positive loadings:")
        for idx in pos_idx:
            if component_loadings[idx] > 0.3:
                print(f"    {metric_names[idx]:25s}: {component_loadings[idx]:+.3f}")

        print("  Strong negative loadings:")
        for idx in neg_idx:
            if component_loadings[idx] < -0.3:
                print(f"    {metric_names[idx]:25s}: {component_loadings[idx]:+.3f}")

        # Attempt interpretation based on metric categories
        high_loading_metrics = [metric_names[j] for j in range(len(metric_names))
                            if abs(component_loadings[j]) > 0.4]

        categories_involved = set()
        for metric in high_loading_metrics:
            for cat, metrics in METRIC_CATEGORIES.items():
                if metric in metrics:
                    categories_involved.add(cat)

        interpretation = f"Represents: {', '.join(categories_involved) if categories_involved else 'Mixed'}"

        interpretations.append({
            'component': f'PC{i+1}',
            'top_metrics': ', '.join([metric_names[j] for j in pos_idx[:2]]),
            'interpretation': interpretation
        })

    return pd.DataFrame(interpretations)


def save_factor_analysis_results(pca_results: Dict, metric_names: List[str],
                                interpretations: pd.DataFrame, output_dir: Path):
    """
    Save factor analysis results to CSV.

    Args:
        pca_results: PCA results dictionary
        metric_names: List of metric names
        interpretations: Component interpretations
        output_dir: Output directory
    """
    # Save explained variance
    variance_df = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(pca_results['explained_variance']))],
        'explained_variance_pct': pca_results['explained_variance'] * 100,
        'cumulative_variance_pct': pca_results['cumulative_variance'] * 100
    })
    variance_df.to_csv(output_dir / 'explained_variance.csv', index=False)

    # Save loadings
    loadings_df = pd.DataFrame(
        pca_results['loadings'].T,
        index=metric_names,
        columns=[f'PC{i+1}' for i in range(pca_results['loadings'].shape[0])]
    )
    loadings_df.to_csv(output_dir / 'factor_loadings.csv')

    # Save interpretations
    interpretations.to_csv(output_dir / 'component_interpretations.csv', index=False)

    print(f"\n✓ Saved factor analysis results to: {output_dir}")
    print(f"  - explained_variance.csv")
    print(f"  - factor_loadings.csv")
    print(f"  - component_interpretations.csv")


def main():
    parser = argparse.ArgumentParser(
        description='Perform factor analysis on diversity metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ablations', 'factorials', 'all'],
        default='ablations',
        help='Dataset to analyze (default: ablations)'
    )

    args = parser.parse_args()

    print("="*80)
    print("METRIC FACTOR ANALYSIS")
    print("Validation 1.3: Phase 4 - Metric Validation")
    print("="*80)
    print()

    # Create output directory
    output_base = Path('results/validation/factor_analysis')
    output_base.mkdir(parents=True, exist_ok=True)

    # Determine datasets to analyze
    datasets = []
    if args.dataset == 'all':
        datasets = ['ablations', 'factorials']
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        try:
            print(f"\n{'='*80}")
            print(f"ANALYZING: {dataset_name.upper()}")
            print(f"{'='*80}")

            # Load data
            df = load_data(dataset_name)

            # Prepare data for PCA
            X, metric_names = prepare_data_for_pca(df)

            # Perform PCA
            pca_results = perform_pca(X, metric_names)

            # Create output directory for this dataset
            output_dir = output_base / dataset_name
            output_dir.mkdir(exist_ok=True)

            # Plot scree plot
            plot_scree_plot(
                pca_results['explained_variance'],
                pca_results['cumulative_variance'],
                output_dir / 'pca_scree_plot.pdf',
                title=f'{dataset_name.capitalize()} - PCA Explained Variance'
            )

            # Plot loadings heatmap
            plot_loadings_heatmap(
                pca_results['loadings'],
                metric_names,
                min(pca_results['n_components_90'], 6),  # Show up to 6 components
                output_dir / 'pca_loadings_heatmap.pdf',
                title=f'{dataset_name.capitalize()} - PCA Loadings'
            )

            # Interpret components
            interpretations = interpret_components(
                pca_results['loadings'],
                metric_names,
                pca_results['n_components_90']
            )

            # Save results
            save_factor_analysis_results(pca_results, metric_names,
                                        interpretations, output_dir)

        except Exception as e:
            print(f"\n✗ Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("FACTOR ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_base}")
    print("\nKey Findings:")
    print("  - Review scree plots to see how many dimensions are needed")
    print("  - Check loadings heatmap to understand what each component represents")
    print("  - Use component_interpretations.csv for qualitative understanding")


if __name__ == '__main__':
    main()
