#!/usr/bin/env python3
"""
Create comprehensive report for capacity ablation study.

This script generates a detailed report with summary statistics and visualizations
for all metrics from the capacity ablation study, grouped by capacity and conditioning.

Usage:
    # Generate comprehensive report for capacity ablation
    python scripts/create_comprehensive_report.py \
        --results_csv results/ablations/capacity/all_results.csv

    # Specify custom output directory
    python scripts/create_comprehensive_report.py \
        --results_csv results/ablations/capacity/all_results.csv \
        --output_dir results/custom_report
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Optional, Dict
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_capacity_results(csv_path: str) -> pd.DataFrame:
    """
    Load capacity ablation results.

    Args:
        csv_path: Path to capacity ablation CSV

    Returns:
        DataFrame with capacity ablation results
    """
    print(f"Loading capacity ablation results from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} experiments")

    # Check required columns
    if 'capacity' not in df.columns:
        raise ValueError("CSV must contain 'capacity' column")
    if 'conditioning' not in df.columns:
        print("  ⚠ Warning: 'conditioning' column not found, some groupings will be limited")

    return df


def create_summary_csvs(df: pd.DataFrame, output_dir: Path):
    """Generate summary CSV files."""
    print("\nGenerating summary CSV files...")

    # Overall summary
    overall_summary = df.describe(include='all')
    overall_summary.to_csv(output_dir / 'overall_summary.csv')
    print(f"  ✓ overall_summary.csv")

    # Numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        # Summary by capacity
        capacity_summary = df.groupby('capacity')[numeric_cols].agg(['mean', 'std', 'count'])
        capacity_summary.columns = ['_'.join(map(str, c)).strip() for c in capacity_summary.columns.values]
        capacity_summary.to_csv(output_dir / 'summary_by_capacity.csv')
        print(f"  ✓ summary_by_capacity.csv")

        # Summary by conditioning (if available)
        if 'conditioning' in df.columns:
            conditioning_summary = df.groupby('conditioning')[numeric_cols].agg(['mean', 'std', 'count'])
            conditioning_summary.columns = ['_'.join(map(str, c)).strip() for c in conditioning_summary.columns.values]
            conditioning_summary.to_csv(output_dir / 'summary_by_conditioning.csv')
            print(f"  ✓ summary_by_conditioning.csv")

            # Summary by capacity and conditioning
            cap_cond_summary = df.groupby(['capacity', 'conditioning'])[numeric_cols].agg(['mean', 'std', 'count'])
            cap_cond_summary.columns = ['_'.join(map(str, c)).strip() for c in cap_cond_summary.columns.values]
            cap_cond_summary.to_csv(output_dir / 'summary_by_capacity_and_conditioning.csv')
            print(f"  ✓ summary_by_capacity_and_conditioning.csv")

    # Save full results
    df.to_csv(output_dir / 'capacity_detailed_results.csv', index=False)
    print(f"  ✓ capacity_detailed_results.csv")


def create_metrics_pdf(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive metrics visualization PDF."""
    pdf_path = output_dir / 'comprehensive_metrics_report.pdf'
    print(f"\nGenerating comprehensive PDF: {pdf_path}")

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Define capacity order
    capacity_order = ['small', 'medium', 'large', 'xlarge']
    available_capacities = [c for c in capacity_order if c in df['capacity'].unique()]

    with PdfPages(pdf_path) as pdf:
        # Page 1: Traditional Metrics by Capacity & Conditioning
        create_traditional_metrics_page(df, pdf, available_capacities)

        # Page 2: Trajectory Metrics (TDS, MPD)
        create_trajectory_metrics_page(df, pdf, available_capacities)

        # Page 3: Spatial Metrics (MCE, PMD, PFS)
        create_spatial_metrics_page(df, pdf, available_capacities)

        # Page 4: Objective & Flow Metrics (PAS, FCI)
        create_objective_flow_metrics_page(df, pdf, available_capacities)

        # Page 5: Dynamics & Composite Metrics (RBD, QDS, DER)
        create_dynamics_composite_metrics_page(df, pdf, available_capacities)

        # Page 6: Performance Metrics
        create_performance_metrics_page(df, pdf, available_capacities)

        # Page 7: Capacity vs Conditioning Heatmaps
        create_heatmap_page(df, pdf, available_capacities)

        # Page 8: Summary Statistics Tables
        create_summary_tables_page(df, pdf, available_capacities)

    print(f"  ✓ PDF created with 8 pages")


def create_traditional_metrics_page(df: pd.DataFrame, pdf: PdfPages, capacity_order: List[str]):
    """Page 1: Traditional MO metrics grouped by capacity and conditioning."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Traditional Multi-Objective Metrics - Capacity Ablation',
                 fontsize=18, fontweight='bold', y=0.995)

    metrics = ['hypervolume', 'r2_indicator', 'avg_pairwise_distance']
    titles = ['Hypervolume', 'R2 Indicator', 'Average Pairwise Distance']

    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()

    for idx, (metric, title) in enumerate(zip(metrics, titles), 1):
        if metric in df.columns:
            # By capacity
            ax1 = plt.subplot(3, 3, (idx-1)*3 + 1)
            df_plot = df[['capacity', metric]].dropna()
            if len(df_plot) > 0:
                sns.boxplot(data=df_plot, x='capacity', y=metric, order=capacity_order, ax=ax1)
                ax1.set_title(f'{title} by Capacity', fontsize=11, fontweight='bold')
                ax1.set_xlabel('Capacity', fontsize=9)
                ax1.set_ylabel(title, fontsize=9)
                plt.xticks(rotation=45, ha='right')

            # By conditioning
            ax2 = plt.subplot(3, 3, (idx-1)*3 + 2)
            if has_conditioning:
                df_plot = df[['conditioning', metric]].dropna()
                if len(df_plot) > 0:
                    sns.boxplot(data=df_plot, x='conditioning', y=metric, ax=ax2)
                    ax2.set_title(f'{title} by Conditioning', fontsize=11, fontweight='bold')
                    ax2.set_xlabel('Conditioning', fontsize=9)
                    ax2.set_ylabel(title, fontsize=9)
                    plt.xticks(rotation=45, ha='right')

            # Grouped bar: capacity x conditioning
            ax3 = plt.subplot(3, 3, (idx-1)*3 + 3)
            if has_conditioning:
                df_grouped = df.groupby(['capacity', 'conditioning'])[metric].mean().unstack()
                df_grouped.reindex(capacity_order).plot(kind='bar', ax=ax3)
                ax3.set_title(f'{title} by Capacity × Conditioning', fontsize=11, fontweight='bold')
                ax3.set_xlabel('Capacity', fontsize=9)
                ax3.set_ylabel(f'Mean {title}', fontsize=9)
                ax3.legend(title='Conditioning', fontsize=7)
                plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_trajectory_metrics_page(df: pd.DataFrame, pdf: PdfPages, capacity_order: List[str]):
    """Page 2: Trajectory metrics (TDS, MPD)."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Trajectory Diversity Metrics - Capacity Ablation',
                 fontsize=18, fontweight='bold', y=0.995)

    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()
    metrics = [('tds', 'Trajectory Diversity Score (TDS)'),
               ('mpd', 'Multi-Path Diversity (MPD)')]

    for idx, (metric, title) in enumerate(metrics, 1):
        if metric in df.columns:
            # Box plot by capacity
            ax1 = plt.subplot(2, 3, (idx-1)*3 + 1)
            df_plot = df[['capacity', metric]].dropna()
            if len(df_plot) > 0:
                sns.boxplot(data=df_plot, x='capacity', y=metric, order=capacity_order, ax=ax1)
                ax1.set_title(f'{title} by Capacity', fontsize=11, fontweight='bold')
                ax1.set_xlabel('Capacity', fontsize=9)
                ax1.set_ylabel(metric.upper(), fontsize=9)
                plt.xticks(rotation=45, ha='right')

            # Box plot by conditioning
            ax2 = plt.subplot(2, 3, (idx-1)*3 + 2)
            if has_conditioning:
                df_plot = df[['conditioning', metric]].dropna()
                if len(df_plot) > 0:
                    sns.boxplot(data=df_plot, x='conditioning', y=metric, ax=ax2)
                    ax2.set_title(f'{title} by Conditioning', fontsize=11, fontweight='bold')
                    ax2.set_xlabel('Conditioning', fontsize=9)
                    ax2.set_ylabel(metric.upper(), fontsize=9)
                    plt.xticks(rotation=45, ha='right')

            # Grouped bar
            ax3 = plt.subplot(2, 3, (idx-1)*3 + 3)
            if has_conditioning:
                df_grouped = df.groupby(['capacity', 'conditioning'])[metric].mean().unstack()
                df_grouped.reindex(capacity_order).plot(kind='bar', ax=ax3)
                ax3.set_title(f'{title} by Capacity × Conditioning', fontsize=11, fontweight='bold')
                ax3.set_xlabel('Capacity', fontsize=9)
                ax3.set_ylabel(f'Mean {metric.upper()}', fontsize=9)
                ax3.legend(title='Conditioning', fontsize=7)
                plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_spatial_metrics_page(df: pd.DataFrame, pdf: PdfPages, capacity_order: List[str]):
    """Page 3: Spatial metrics (MCE, PMD, PFS)."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Spatial Diversity Metrics - Capacity Ablation',
                 fontsize=18, fontweight='bold', y=0.995)

    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()
    metrics = [('mce', 'Mode Coverage Entropy (MCE)'),
               ('pmd', 'Pairwise Minimum Distance (PMD)'),
               ('pfs', 'Pareto Front Smoothness (PFS)')]

    for idx, (metric, title) in enumerate(metrics, 1):
        if metric in df.columns:
            # By capacity
            ax1 = plt.subplot(3, 3, (idx-1)*3 + 1)
            df_plot = df[['capacity', metric]].dropna()
            if len(df_plot) > 0:
                sns.boxplot(data=df_plot, x='capacity', y=metric, order=capacity_order, ax=ax1)
                ax1.set_title(f'{title} by Capacity', fontsize=10, fontweight='bold')
                ax1.set_xlabel('Capacity', fontsize=8)
                ax1.set_ylabel(metric.upper(), fontsize=8)
                plt.xticks(rotation=45, ha='right')

            # By conditioning
            ax2 = plt.subplot(3, 3, (idx-1)*3 + 2)
            if has_conditioning:
                df_plot = df[['conditioning', metric]].dropna()
                if len(df_plot) > 0:
                    sns.boxplot(data=df_plot, x='conditioning', y=metric, ax=ax2)
                    ax2.set_title(f'{title} by Conditioning', fontsize=10, fontweight='bold')
                    ax2.set_xlabel('Conditioning', fontsize=8)
                    ax2.set_ylabel(metric.upper(), fontsize=8)
                    plt.xticks(rotation=45, ha='right')

            # Grouped
            ax3 = plt.subplot(3, 3, (idx-1)*3 + 3)
            if has_conditioning:
                df_grouped = df.groupby(['capacity', 'conditioning'])[metric].mean().unstack()
                df_grouped.reindex(capacity_order).plot(kind='bar', ax=ax3)
                ax3.set_title(f'{title} by Capacity × Conditioning', fontsize=10, fontweight='bold')
                ax3.set_xlabel('Capacity', fontsize=8)
                ax3.set_ylabel(f'Mean {metric.upper()}', fontsize=8)
                ax3.legend(title='Conditioning', fontsize=6)
                plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_objective_flow_metrics_page(df: pd.DataFrame, pdf: PdfPages, capacity_order: List[str]):
    """Page 4: Objective (PAS) and Flow (FCI) metrics."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Objective & Flow Metrics - Capacity Ablation',
                 fontsize=18, fontweight='bold', y=0.995)

    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()
    metrics = [('pas', 'Preference-Aligned Spread (PAS)'),
               ('fci', 'Flow Concentration Index (FCI)')]

    for idx, (metric, title) in enumerate(metrics, 1):
        if metric in df.columns:
            # By capacity
            ax1 = plt.subplot(2, 3, (idx-1)*3 + 1)
            df_plot = df[['capacity', metric]].dropna()
            if len(df_plot) > 0:
                sns.boxplot(data=df_plot, x='capacity', y=metric, order=capacity_order, ax=ax1)
                ax1.set_title(f'{title} by Capacity', fontsize=11, fontweight='bold')
                ax1.set_xlabel('Capacity', fontsize=9)
                ax1.set_ylabel(metric.upper(), fontsize=9)
                plt.xticks(rotation=45, ha='right')

            # By conditioning
            ax2 = plt.subplot(2, 3, (idx-1)*3 + 2)
            if has_conditioning:
                df_plot = df[['conditioning', metric]].dropna()
                if len(df_plot) > 0:
                    sns.boxplot(data=df_plot, x='conditioning', y=metric, ax=ax2)
                    ax2.set_title(f'{title} by Conditioning', fontsize=11, fontweight='bold')
                    ax2.set_xlabel('Conditioning', fontsize=9)
                    ax2.set_ylabel(metric.upper(), fontsize=9)
                    plt.xticks(rotation=45, ha='right')

            # Grouped
            ax3 = plt.subplot(2, 3, (idx-1)*3 + 3)
            if has_conditioning:
                df_grouped = df.groupby(['capacity', 'conditioning'])[metric].mean().unstack()
                df_grouped.reindex(capacity_order).plot(kind='bar', ax=ax3)
                ax3.set_title(f'{title} by Capacity × Conditioning', fontsize=11, fontweight='bold')
                ax3.set_xlabel('Capacity', fontsize=9)
                ax3.set_ylabel(f'Mean {metric.upper()}', fontsize=9)
                ax3.legend(title='Conditioning', fontsize=7)
                plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_dynamics_composite_metrics_page(df: pd.DataFrame, pdf: PdfPages, capacity_order: List[str]):
    """Page 5: Dynamics (RBD) and Composite (QDS, DER) metrics."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Dynamics & Composite Metrics - Capacity Ablation',
                 fontsize=18, fontweight='bold', y=0.995)

    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()
    metrics = [('rbd', 'Replay Buffer Diversity (RBD)'),
               ('qds', 'Quality-Diversity Score (QDS)'),
               ('der', 'Diversity-Efficiency Ratio (DER)')]

    for idx, (metric, title) in enumerate(metrics, 1):
        if metric in df.columns:
            # By capacity
            ax1 = plt.subplot(3, 3, (idx-1)*3 + 1)
            df_plot = df[['capacity', metric]].dropna()
            if len(df_plot) > 0:
                sns.boxplot(data=df_plot, x='capacity', y=metric, order=capacity_order, ax=ax1)
                ax1.set_title(f'{title} by Capacity', fontsize=10, fontweight='bold')
                ax1.set_xlabel('Capacity', fontsize=8)
                ax1.set_ylabel(metric.upper(), fontsize=8)
                plt.xticks(rotation=45, ha='right')

            # By conditioning
            ax2 = plt.subplot(3, 3, (idx-1)*3 + 2)
            if has_conditioning:
                df_plot = df[['conditioning', metric]].dropna()
                if len(df_plot) > 0:
                    sns.boxplot(data=df_plot, x='conditioning', y=metric, ax=ax2)
                    ax2.set_title(f'{title} by Conditioning', fontsize=10, fontweight='bold')
                    ax2.set_xlabel('Conditioning', fontsize=8)
                    ax2.set_ylabel(metric.upper(), fontsize=8)
                    plt.xticks(rotation=45, ha='right')

            # Grouped
            ax3 = plt.subplot(3, 3, (idx-1)*3 + 3)
            if has_conditioning:
                df_grouped = df.groupby(['capacity', 'conditioning'])[metric].mean().unstack()
                df_grouped.reindex(capacity_order).plot(kind='bar', ax=ax3)
                ax3.set_title(f'{title} by Capacity × Conditioning', fontsize=10, fontweight='bold')
                ax3.set_xlabel('Capacity', fontsize=8)
                ax3.set_ylabel(f'Mean {metric.upper()}', fontsize=8)
                ax3.legend(title='Conditioning', fontsize=6)
                plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_performance_metrics_page(df: pd.DataFrame, pdf: PdfPages, capacity_order: List[str]):
    """Page 6: Performance metrics (training time, parameters, loss)."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Performance & Computational Metrics - Capacity Ablation',
                 fontsize=18, fontweight='bold', y=0.995)

    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()

    # Training time
    if 'training_time' in df.columns:
        ax1 = plt.subplot(3, 3, 1)
        df_plot = df[['capacity', 'training_time']].dropna()
        if len(df_plot) > 0:
            sns.boxplot(data=df_plot, x='capacity', y='training_time', order=capacity_order, ax=ax1)
            ax1.set_title('Training Time by Capacity', fontsize=10, fontweight='bold')
            ax1.set_xlabel('Capacity', fontsize=8)
            ax1.set_ylabel('Time (s)', fontsize=8)
            plt.xticks(rotation=45, ha='right')

        if has_conditioning:
            ax2 = plt.subplot(3, 3, 2)
            df_plot = df[['conditioning', 'training_time']].dropna()
            if len(df_plot) > 0:
                sns.boxplot(data=df_plot, x='conditioning', y='training_time', ax=ax2)
                ax2.set_title('Training Time by Conditioning', fontsize=10, fontweight='bold')
                ax2.set_xlabel('Conditioning', fontsize=8)
                ax2.set_ylabel('Time (s)', fontsize=8)
                plt.xticks(rotation=45, ha='right')

            ax3 = plt.subplot(3, 3, 3)
            df_grouped = df.groupby(['capacity', 'conditioning'])['training_time'].mean().unstack()
            df_grouped.reindex(capacity_order).plot(kind='bar', ax=ax3)
            ax3.set_title('Training Time by Capacity × Conditioning', fontsize=10, fontweight='bold')
            ax3.set_xlabel('Capacity', fontsize=8)
            ax3.set_ylabel('Mean Time (s)', fontsize=8)
            ax3.legend(title='Conditioning', fontsize=6)
            plt.xticks(rotation=45, ha='right')

    # Number of parameters
    if 'num_parameters' in df.columns:
        ax4 = plt.subplot(3, 3, 4)
        df_plot = df[['capacity', 'num_parameters']].dropna()
        if len(df_plot) > 0:
            sns.boxplot(data=df_plot, x='capacity', y='num_parameters', order=capacity_order, ax=ax4)
            ax4.set_title('Model Parameters by Capacity', fontsize=10, fontweight='bold')
            ax4.set_xlabel('Capacity', fontsize=8)
            ax4.set_ylabel('# Parameters', fontsize=8)
            plt.xticks(rotation=45, ha='right')

        if has_conditioning:
            ax5 = plt.subplot(3, 3, 5)
            df_plot = df[['conditioning', 'num_parameters']].dropna()
            if len(df_plot) > 0:
                sns.boxplot(data=df_plot, x='conditioning', y='num_parameters', ax=ax5)
                ax5.set_title('Model Parameters by Conditioning', fontsize=10, fontweight='bold')
                ax5.set_xlabel('Conditioning', fontsize=8)
                ax5.set_ylabel('# Parameters', fontsize=8)
                plt.xticks(rotation=45, ha='right')

            ax6 = plt.subplot(3, 3, 6)
            df_grouped = df.groupby(['capacity', 'conditioning'])['num_parameters'].mean().unstack()
            df_grouped.reindex(capacity_order).plot(kind='bar', ax=ax6)
            ax6.set_title('Parameters by Capacity × Conditioning', fontsize=10, fontweight='bold')
            ax6.set_xlabel('Capacity', fontsize=8)
            ax6.set_ylabel('Mean # Parameters', fontsize=8)
            ax6.legend(title='Conditioning', fontsize=6)
            plt.xticks(rotation=45, ha='right')

    # Final loss
    if 'final_loss' in df.columns:
        ax7 = plt.subplot(3, 3, 7)
        df_plot = df[['capacity', 'final_loss']].dropna()
        if len(df_plot) > 0:
            sns.boxplot(data=df_plot, x='capacity', y='final_loss', order=capacity_order, ax=ax7)
            ax7.set_title('Final Loss by Capacity', fontsize=10, fontweight='bold')
            ax7.set_xlabel('Capacity', fontsize=8)
            ax7.set_ylabel('Loss', fontsize=8)
            plt.xticks(rotation=45, ha='right')

        if has_conditioning:
            ax8 = plt.subplot(3, 3, 8)
            df_plot = df[['conditioning', 'final_loss']].dropna()
            if len(df_plot) > 0:
                sns.boxplot(data=df_plot, x='conditioning', y='final_loss', ax=ax8)
                ax8.set_title('Final Loss by Conditioning', fontsize=10, fontweight='bold')
                ax8.set_xlabel('Conditioning', fontsize=8)
                ax8.set_ylabel('Loss', fontsize=8)
                plt.xticks(rotation=45, ha='right')

            ax9 = plt.subplot(3, 3, 9)
            df_grouped = df.groupby(['capacity', 'conditioning'])['final_loss'].mean().unstack()
            df_grouped.reindex(capacity_order).plot(kind='bar', ax=ax9)
            ax9.set_title('Loss by Capacity × Conditioning', fontsize=10, fontweight='bold')
            ax9.set_xlabel('Capacity', fontsize=8)
            ax9.set_ylabel('Mean Loss', fontsize=8)
            ax9.legend(title='Conditioning', fontsize=6)
            plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_heatmap_page(df: pd.DataFrame, pdf: PdfPages, capacity_order: List[str]):
    """Page 7: Heatmaps showing capacity vs conditioning for all metrics."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Capacity × Conditioning Heatmaps',
                 fontsize=18, fontweight='bold', y=0.995)

    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()

    if has_conditioning:
        # Select key metrics for heatmaps
        key_metrics = ['hypervolume', 'tds', 'mce', 'pas', 'fci', 'qds']
        available_metrics = [m for m in key_metrics if m in df.columns]

        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        for idx, metric in enumerate(available_metrics, 1):
            ax = plt.subplot(n_rows, n_cols, idx)

            # Create pivot table
            pivot_data = df.pivot_table(values=metric,
                                       index='capacity',
                                       columns='conditioning',
                                       aggfunc='mean')

            # Reorder by capacity
            pivot_data = pivot_data.reindex(capacity_order)

            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd',
                       ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
            ax.set_xlabel('Conditioning', fontsize=9)
            ax.set_ylabel('Capacity', fontsize=9)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)

    else:
        # If no conditioning, show capacity comparison for key metrics
        key_metrics = ['hypervolume', 'tds', 'mce', 'pas', 'fci', 'qds']
        available_metrics = [m for m in key_metrics if m in df.columns]

        ax = plt.subplot(1, 1, 1)
        summary_data = df.groupby('capacity')[available_metrics].mean()
        summary_data = summary_data.reindex(capacity_order)

        sns.heatmap(summary_data.T, annot=True, fmt='.4f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Mean Metric Values by Capacity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Capacity', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_summary_tables_page(df: pd.DataFrame, pdf: PdfPages, capacity_order: List[str]):
    """Page 8: Summary statistics tables."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Summary Statistics Tables - Capacity Ablation',
                 fontsize=18, fontweight='bold', y=0.995)

    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()

    # Key metrics for summary
    key_metrics = ['hypervolume', 'tds', 'mce', 'pas', 'fci', 'qds', 'der']
    available_metrics = [m for m in key_metrics if m in df.columns]

    # Table 1: By Capacity
    ax1 = plt.subplot(3, 1, 1)
    ax1.axis('tight')
    ax1.axis('off')

    stats_data = []
    for capacity in capacity_order:
        df_sub = df[df['capacity'] == capacity]
        row = [capacity]
        for metric in available_metrics:
            data = df_sub[metric].dropna()
            if len(data) > 0:
                row.append(f'{data.mean():.3f}±{data.std():.3f}')
            else:
                row.append('N/A')
        stats_data.append(row)

    if stats_data:
        col_labels = ['Capacity'] + [m.upper() for m in available_metrics]
        table = ax1.table(cellText=stats_data, colLabels=col_labels,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        # Style header
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax1.set_title('Mean ± Std by Capacity', fontsize=12, fontweight='bold', pad=20)

    # Table 2: By Conditioning (if available)
    if has_conditioning:
        ax2 = plt.subplot(3, 1, 2)
        ax2.axis('tight')
        ax2.axis('off')

        stats_data = []
        for conditioning in df['conditioning'].unique():
            df_sub = df[df['conditioning'] == conditioning]
            row = [conditioning]
            for metric in available_metrics:
                data = df_sub[metric].dropna()
                if len(data) > 0:
                    row.append(f'{data.mean():.3f}±{data.std():.3f}')
                else:
                    row.append('N/A')
            stats_data.append(row)

        if stats_data:
            col_labels = ['Conditioning'] + [m.upper() for m in available_metrics]
            table = ax2.table(cellText=stats_data, colLabels=col_labels,
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)

            # Style header
            for i in range(len(col_labels)):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')

            ax2.set_title('Mean ± Std by Conditioning', fontsize=12, fontweight='bold', pad=20)

    # Table 3: Overall statistics
    ax3 = plt.subplot(3, 1, 3)
    ax3.axis('tight')
    ax3.axis('off')

    stats_data = []
    for metric in available_metrics:
        data = df[metric].dropna()
        if len(data) > 0:
            stats_data.append([
                metric.upper(),
                f'{data.mean():.4f}',
                f'{data.std():.4f}',
                f'{data.min():.4f}',
                f'{data.max():.4f}',
                f'{len(data)}'
            ])

    if stats_data:
        table = ax3.table(cellText=stats_data,
                         colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max', 'Count'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax3.set_title('Overall Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive report for capacity ablation study'
    )
    parser.add_argument(
        '--results_csv',
        type=str,
        required=True,
        help='Path to capacity ablation results CSV (e.g., results/ablations/capacity/all_results.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/ablations/reports',
        help='Output directory for comprehensive report'
    )

    args = parser.parse_args()

    print("="*70)
    print("CAPACITY ABLATION COMPREHENSIVE REPORT GENERATOR")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load results
    df = load_capacity_results(args.results_csv)

    # Generate summary CSVs
    create_summary_csvs(df, output_dir)

    # Generate comprehensive PDF
    create_metrics_pdf(df, output_dir)

    print("\n" + "="*70)
    print("✓ COMPREHENSIVE REPORT GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in {output_dir}:")
    print("  - comprehensive_metrics_report.pdf (8-page visualization)")
    print("  - overall_summary.csv")
    print("  - summary_by_capacity.csv")
    print("  - summary_by_conditioning.csv")
    print("  - summary_by_capacity_and_conditioning.csv")
    print("  - capacity_detailed_results.csv")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
