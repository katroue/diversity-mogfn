#!/usr/bin/env python3
"""
Create metrics comparison PDF for ablation study results.

This script generates visualizations comparing FCI (Flow Concentration Index)
and PAS (Preference-Aligned Spread) metrics across different experimental conditions.

Usage:
    # For capacity ablation
    python scripts/create_metrics_comparison_pdf.py \
        --results_csv results/ablations/capacity/all_results.csv \
        --ablation capacity \
        --output_dir results/ablations/capacity/report

    # For sampling ablation
    python scripts/create_metrics_comparison_pdf.py \
        --results_csv results/ablations/sampling/all_results.csv \
        --ablation sampling \
        --output_dir results/ablations/sampling/report

    # For loss ablation
    python scripts/create_metrics_comparison_pdf.py \
        --results_csv results/ablations/loss/all_results.csv \
        --ablation loss \
        --output_dir results/ablations/loss/report
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_metrics_comparison_pdf(results_csv: str,
                                  ablation_type: str,
                                  output_dir: str):
    """
    Create a PDF with metrics comparison plots.

    Args:
        results_csv: Path to the results CSV file
        ablation_type: Type of ablation study ('capacity', 'sampling', 'loss')
        output_dir: Directory to save the output PDF
    """
    # Load results
    print(f"Loading results from {results_csv}...")
    df = pd.read_csv(results_csv)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create PDF
    pdf_path = output_dir / 'metrics_comparison.pdf'
    print(f"Creating PDF at {pdf_path}...")

    with PdfPages(pdf_path) as pdf:
        # Page 1: FCI and PAS by main experimental variable
        create_main_comparison_page(df, ablation_type, pdf)

        # Page 2: Detailed breakdown
        create_detailed_breakdown_page(df, ablation_type, pdf)

        # Page 3: Correlation and statistics
        create_statistics_page(df, ablation_type, pdf)

    print(f"✓ PDF created successfully: {pdf_path}")


def create_main_comparison_page(df: pd.DataFrame, ablation_type: str, pdf: PdfPages):
    """Create main comparison page with FCI and PAS metrics."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{ablation_type.title()} Ablation: FCI and PAS Metrics',
                 fontsize=16, fontweight='bold')

    # Determine grouping variable
    if ablation_type == 'capacity':
        group_var = 'capacity'
        group_order = ['small', 'medium', 'large', 'xlarge']
    elif ablation_type == 'sampling':
        group_var = 'preference_sampling'
        group_order = None  # Will be determined from data
    elif ablation_type == 'loss':
        group_var = 'loss'
        group_order = None
    else:
        group_var = 'exp_name'
        group_order = None

    # Filter out missing values
    df_plot = df[[group_var, 'fci', 'pas']].dropna()

    # Subplot 1: FCI comparison
    ax1 = plt.subplot(2, 2, 1)
    if group_order:
        sns.boxplot(data=df_plot, x=group_var, y='fci', order=group_order, ax=ax1)
    else:
        sns.boxplot(data=df_plot, x=group_var, y='fci', ax=ax1)
    ax1.set_title('Flow Concentration Index (FCI)', fontsize=12, fontweight='bold')
    ax1.set_xlabel(group_var.replace('_', ' ').title(), fontsize=10)
    ax1.set_ylabel('FCI', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    # Subplot 2: PAS comparison
    ax2 = plt.subplot(2, 2, 2)
    if group_order:
        sns.boxplot(data=df_plot, x=group_var, y='pas', order=group_order, ax=ax2)
    else:
        sns.boxplot(data=df_plot, x=group_var, y='pas', ax=ax2)
    ax2.set_title('Preference-Aligned Spread (PAS)', fontsize=12, fontweight='bold')
    ax2.set_xlabel(group_var.replace('_', ' ').title(), fontsize=10)
    ax2.set_ylabel('PAS', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    # Subplot 3: FCI with mean markers
    ax3 = plt.subplot(2, 2, 3)
    if group_order:
        sns.violinplot(data=df_plot, x=group_var, y='fci', order=group_order, ax=ax3, inner='quartile')
    else:
        sns.violinplot(data=df_plot, x=group_var, y='fci', ax=ax3, inner='quartile')
    ax3.set_title('FCI Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel(group_var.replace('_', ' ').title(), fontsize=10)
    ax3.set_ylabel('FCI', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    # Subplot 4: PAS with mean markers
    ax4 = plt.subplot(2, 2, 4)
    if group_order:
        sns.violinplot(data=df_plot, x=group_var, y='pas', order=group_order, ax=ax4, inner='quartile')
    else:
        sns.violinplot(data=df_plot, x=group_var, y='pas', ax=ax4, inner='quartile')
    ax4.set_title('PAS Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel(group_var.replace('_', ' ').title(), fontsize=10)
    ax4.set_ylabel('PAS', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


def create_detailed_breakdown_page(df: pd.DataFrame, ablation_type: str, pdf: PdfPages):
    """Create detailed breakdown page with multiple groupings."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{ablation_type.title()} Ablation: Detailed Breakdown',
                 fontsize=16, fontweight='bold')

    # Determine if we have secondary grouping
    has_conditioning = 'conditioning' in df.columns and df['conditioning'].notna().any()

    if ablation_type == 'capacity' and has_conditioning:
        # Group by capacity and conditioning
        ax1 = plt.subplot(2, 2, 1)
        df_grouped = df.groupby(['capacity', 'conditioning'])['fci'].mean().unstack()
        df_grouped.plot(kind='bar', ax=ax1)
        ax1.set_title('FCI by Capacity and Conditioning', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Capacity', fontsize=10)
        ax1.set_ylabel('Mean FCI', fontsize=10)
        ax1.legend(title='Conditioning', fontsize=8)
        plt.xticks(rotation=45, ha='right')

        ax2 = plt.subplot(2, 2, 2)
        df_grouped = df.groupby(['capacity', 'conditioning'])['pas'].mean().unstack()
        df_grouped.plot(kind='bar', ax=ax2)
        ax2.set_title('PAS by Capacity and Conditioning', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Capacity', fontsize=10)
        ax2.set_ylabel('Mean PAS', fontsize=10)
        ax2.legend(title='Conditioning', fontsize=8)
        plt.xticks(rotation=45, ha='right')
    else:
        # Simple grouped plots
        ax1 = plt.subplot(2, 2, 1)
        df.boxplot(column='fci', ax=ax1)
        ax1.set_title('Overall FCI Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('FCI', fontsize=10)

        ax2 = plt.subplot(2, 2, 2)
        df.boxplot(column='pas', ax=ax2)
        ax2.set_title('Overall PAS Distribution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('PAS', fontsize=10)

    # Subplot 3: FCI vs training time
    ax3 = plt.subplot(2, 2, 3)
    if 'training_time' in df.columns:
        scatter = ax3.scatter(df['training_time'], df['fci'],
                            c=df['num_parameters'] if 'num_parameters' in df.columns else None,
                            cmap='viridis', alpha=0.6)
        ax3.set_title('FCI vs Training Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Training Time (s)', fontsize=10)
        ax3.set_ylabel('FCI', fontsize=10)
        if 'num_parameters' in df.columns:
            plt.colorbar(scatter, ax=ax3, label='# Parameters')

    # Subplot 4: PAS vs training time
    ax4 = plt.subplot(2, 2, 4)
    if 'training_time' in df.columns:
        scatter = ax4.scatter(df['training_time'], df['pas'],
                            c=df['num_parameters'] if 'num_parameters' in df.columns else None,
                            cmap='viridis', alpha=0.6)
        ax4.set_title('PAS vs Training Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Training Time (s)', fontsize=10)
        ax4.set_ylabel('PAS', fontsize=10)
        if 'num_parameters' in df.columns:
            plt.colorbar(scatter, ax=ax4, label='# Parameters')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


def create_statistics_page(df: pd.DataFrame, ablation_type: str, pdf: PdfPages):
    """Create statistics and correlation page."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{ablation_type.title()} Ablation: Statistics and Correlations',
                 fontsize=16, fontweight='bold')

    # Subplot 1: Correlation heatmap
    ax1 = plt.subplot(2, 2, 1)

    # Select numeric columns of interest
    metrics_cols = ['fci', 'pas', 'hypervolume', 'tds', 'mce', 'qds']
    available_cols = [col for col in metrics_cols if col in df.columns]

    if len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax1, square=True)
        ax1.set_title('Metric Correlations', fontsize=12, fontweight='bold')

    # Subplot 2: FCI vs PAS scatter
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(df['fci'], df['pas'], alpha=0.6)
    ax2.set_title('FCI vs PAS', fontsize=12, fontweight='bold')
    ax2.set_xlabel('FCI', fontsize=10)
    ax2.set_ylabel('PAS', fontsize=10)

    # Add correlation coefficient
    if df['fci'].notna().any() and df['pas'].notna().any():
        corr = df[['fci', 'pas']].corr().iloc[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 3: Summary statistics table for FCI
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('tight')
    ax3.axis('off')

    fci_stats = df['fci'].describe()
    table_data = [[f'{stat}', f'{fci_stats[stat]:.4f}'] for stat in fci_stats.index]
    table = ax3.table(cellText=table_data, colLabels=['Statistic', 'FCI'],
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax3.set_title('FCI Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # Subplot 4: Summary statistics table for PAS
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('tight')
    ax4.axis('off')

    pas_stats = df['pas'].describe()
    table_data = [[f'{stat}', f'{pas_stats[stat]:.4f}'] for stat in pas_stats.index]
    table = ax4.table(cellText=table_data, colLabels=['Statistic', 'PAS'],
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('PAS Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create metrics comparison PDF with FCI and PAS visualizations'
    )
    parser.add_argument(
        '--results_csv',
        type=str,
        required=True,
        help='Path to results CSV file (e.g., results/ablations/capacity/all_results.csv)'
    )
    parser.add_argument(
        '--ablation',
        type=str,
        required=True,
        choices=['capacity', 'sampling', 'loss', 'custom'],
        help='Type of ablation study'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/ablations/report',
        help='Output directory for PDF file'
    )

    args = parser.parse_args()

    create_metrics_comparison_pdf(
        results_csv=args.results_csv,
        ablation_type=args.ablation,
        output_dir=args.output_dir
    )

    print("\n✓ Metrics comparison PDF created successfully!")


if __name__ == '__main__':
    main()
