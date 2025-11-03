#!/usr/bin/env python3
"""
Entropy Coefficient Comparison - Diversity Metrics Visualization

This script creates a comprehensive comparison of different entropy coefficient
values and their impact on diversity metrics in Multi-Objective GFlowNets.

Usage:
    python scripts/ablations/plot_entropy_comparison.py --input results/ablations/loss/entropy_regularization/summary.csv 
    --output results/ablations/loss/entropy_regularization/entropy_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def load_data(csv_path):
    """Load the summary CSV file."""
    return pd.read_csv(csv_path)


def create_comparison_plot(df, output_path='entropy_comparison.png'):
    """
    Create comprehensive 6-panel comparison figure.
    
    Args:
        df: DataFrame with experimental results
        output_path: Path to save the figure
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Entropy Coefficient Impact on Diversity Metrics', 
                 fontsize=16, fontweight='bold')
    
    configs = df['configuration'].values
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # ========================================================================
    # Panel 1: Mode Coverage Entropy (Primary diversity metric)
    # ========================================================================
    ax1 = axes[0, 0]
    mce = df['mode_coverage_entropy_mean'].values
    mce_std = df['mode_coverage_entropy_std'].values
    
    bars1 = ax1.bar(range(len(configs)), mce, yerr=mce_std, 
                    capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.set_ylabel('Mode Coverage Entropy', fontweight='bold', fontsize=11)
    ax1.set_title('(a) Mode Coverage Entropy (MCE)\n↑ Higher = More Modes', 
                  fontweight='bold', fontsize=12)
    ax1.axhline(y=mce[0], color='red', linestyle='--', alpha=0.5, 
                linewidth=2, label='Baseline (no entropy)')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Annotate best
    best_idx = np.argmax(mce)
    ax1.text(best_idx, mce[best_idx] + mce_std[best_idx] + 0.02, '★ Best MCE', 
             ha='center', fontweight='bold', color='green', fontsize=10)
    
    # ========================================================================
    # Panel 2: Avg Pairwise Distance
    # ========================================================================
    ax2 = axes[0, 1]
    dist = df['avg_pairwise_distance_mean'].values
    dist_std = df['avg_pairwise_distance_std'].values
    
    bars2 = ax2.bar(range(len(configs)), dist, yerr=dist_std, 
                    capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.set_ylabel('Avg Pairwise Distance', fontweight='bold', fontsize=11)
    ax2.set_title('(b) Avg Pairwise Distance\n↑ Higher = More Spread', 
                  fontweight='bold', fontsize=12)
    ax2.axhline(y=dist[0], color='red', linestyle='--', alpha=0.5, 
                linewidth=2, label='Baseline')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Annotate best
    best_idx = np.argmax(dist)
    ax2.text(best_idx, dist[best_idx] + dist_std[best_idx] + 0.01, '★', 
             ha='center', fontweight='bold', color='green', fontsize=16)
    
    # ========================================================================
    # Panel 3: Flow Concentration Index
    # ========================================================================
    ax3 = axes[0, 2]
    fci = df['fci_mean'].values
    fci_std = df['fci_std'].values
    
    bars3 = ax3.bar(range(len(configs)), fci, yerr=fci_std, 
                    capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=45, ha='right')
    ax3.set_ylabel('Flow Concentration Index', fontweight='bold', fontsize=11)
    ax3.set_title('(c) Flow Concentration (FCI)\n↓ Lower = Better Distribution', 
                  fontweight='bold', fontsize=12)
    ax3.axhline(y=fci[0], color='red', linestyle='--', alpha=0.5, 
                linewidth=2, label='Baseline')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # Annotate best (lowest FCI)
    best_idx = np.argmin(fci)
    ax3.text(best_idx, fci[best_idx] - fci_std[best_idx] - 0.015, '★ Best FCI', 
             ha='center', fontweight='bold', color='green', fontsize=10)
    
    # ========================================================================
    # Panel 4: Trajectory Diversity Score
    # ========================================================================
    ax4 = axes[1, 0]
    tds = df['tds_mean'].values
    tds_std = df['tds_std'].values
    
    bars4 = ax4.bar(range(len(configs)), tds, yerr=tds_std, 
                    capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(len(configs)))
    ax4.set_xticklabels(configs, rotation=45, ha='right')
    ax4.set_ylabel('Trajectory Diversity Score', fontweight='bold', fontsize=11)
    ax4.set_title('(d) Trajectory Diversity (TDS)\n↑ Higher = More Diverse Paths', 
                  fontweight='bold', fontsize=12)
    ax4.axhline(y=tds[0], color='red', linestyle='--', alpha=0.5, 
                linewidth=2, label='Baseline')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # ========================================================================
    # Panel 5: Hypervolume (Quality check)
    # ========================================================================
    ax5 = axes[1, 1]
    hv = df['hypervolume_mean'].values
    hv_std = df['hypervolume_std'].values
    
    bars5 = ax5.bar(range(len(configs)), hv, yerr=hv_std, 
                    capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_xticks(range(len(configs)))
    ax5.set_xticklabels(configs, rotation=45, ha='right')
    ax5.set_ylabel('Hypervolume', fontweight='bold', fontsize=11)
    ax5.set_title('(e) Hypervolume (Quality)\n↑ Higher = Better Pareto Front', 
                  fontweight='bold', fontsize=12)
    ax5.axhline(y=hv[0], color='red', linestyle='--', alpha=0.5, 
                linewidth=2, label='Baseline')
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    
    # ========================================================================
    # Panel 6: Overall Ranking (composite view)
    # ========================================================================
    ax6 = axes[1, 2]
    
    # Calculate average rank across key diversity metrics
    # Lower rank = better performance
    avg_ranks = []
    for config in configs:
        ranks = []
        
        # MCE rank (higher better, so count how many are >= this value)
        mce_val = df[df['configuration']==config]['mode_coverage_entropy_mean'].values[0]
        mce_rank = (df['mode_coverage_entropy_mean'] >= mce_val).sum()
        ranks.append(mce_rank)
        
        # Distance rank (higher better)
        dist_val = df[df['configuration']==config]['avg_pairwise_distance_mean'].values[0]
        dist_rank = (df['avg_pairwise_distance_mean'] >= dist_val).sum()
        ranks.append(dist_rank)
        
        # FCI rank (lower better, so count how many are <= this value)
        fci_val = df[df['configuration']==config]['fci_mean'].values[0]
        fci_rank = (df['fci_mean'] <= fci_val).sum()
        ranks.append(fci_rank)
        
        # TDS rank (higher better)
        tds_val = df[df['configuration']==config]['tds_mean'].values[0]
        tds_rank = (df['tds_mean'] >= tds_val).sum()
        ranks.append(tds_rank)
        
        avg_ranks.append(np.mean(ranks))
    
    # Create horizontal bar chart
    bars6 = ax6.barh(range(len(configs)), avg_ranks, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
    ax6.set_yticks(range(len(configs)))
    ax6.set_yticklabels(configs)
    ax6.set_xlabel('Average Rank (lower = better)', fontweight='bold', fontsize=11)
    ax6.set_title('(f) Overall Ranking\nAcross Key Diversity Metrics', 
                  fontweight='bold', fontsize=12)
    ax6.invert_xaxis()  # Lower ranks on right (better)
    ax6.grid(axis='x', alpha=0.3)
    
    # Annotate best
    best_idx = np.argmin(avg_ranks)
    ax6.text(avg_ranks[best_idx] - 0.15, best_idx, '★ BEST OVERALL', 
             ha='right', va='center', fontweight='bold', color='green', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    
    # ========================================================================
    # Final adjustments and save
    # ========================================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_path}")
    
    return fig, axes


def print_summary_table(df):
    """Print summary statistics table."""
    configs = df['configuration'].values
    
    print("\n" + "="*80)
    print("SUMMARY RECOMMENDATION TABLE")
    print("="*80)
    
    # Calculate ranks
    summary = pd.DataFrame({
        'Entropy Coef': configs,
        'MCE Rank': df['mode_coverage_entropy_mean'].rank(ascending=False).astype(int),
        'Distance Rank': df['avg_pairwise_distance_mean'].rank(ascending=False).astype(int),
        'FCI Rank': df['fci_mean'].rank(ascending=True).astype(int),
        'TDS Rank': df['tds_mean'].rank(ascending=False).astype(int),
    })
    
    # Calculate overall rank (average of individual ranks)
    summary['Overall Rank'] = summary[['MCE Rank', 'Distance Rank', 'FCI Rank', 'TDS Rank']].mean(axis=1).round(2)
    summary = summary.sort_values('Overall Rank')
    
    print(summary.to_string(index=False))
    print("\n" + "="*80)
    
    # Print best configuration details
    best_config = summary.iloc[0]['Entropy Coef']
    best_row = df[df['configuration'] == best_config].iloc[0]
    
    print(f"\n🏆 BEST CONFIGURATION: {best_config}")
    print(f"   Overall Rank Score: {summary.iloc[0]['Overall Rank']}")
    print(f"\n   Key Diversity Metrics:")
    print(f"   • Mode Coverage Entropy: {best_row['mode_coverage_entropy_mean']:.4f} ± {best_row['mode_coverage_entropy_std']:.4f}")
    print(f"   • Avg Pairwise Distance: {best_row['avg_pairwise_distance_mean']:.4f} ± {best_row['avg_pairwise_distance_std']:.4f}")
    print(f"   • Flow Concentration:    {best_row['fci_mean']:.4f} ± {best_row['fci_std']:.4f} (lower = better)")
    print(f"   • Trajectory Diversity:  {best_row['tds_mean']:.4f} ± {best_row['tds_std']:.4f}")
    print(f"   • Hypervolume (Quality): {best_row['hypervolume_mean']:.4f} ± {best_row['hypervolume_std']:.4f}")
    print("\n" + "="*80)


def main():
    """Main function to generate entropy comparison figure."""
    parser = argparse.ArgumentParser(
        description='Generate entropy coefficient comparison figure'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='summary.csv',
        help='Path to input CSV file (default: summary.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='entropy_comparison.png',
        help='Path to output figure (default: entropy_comparison.png)'
    )
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='Show the plot interactively (in addition to saving)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input}")
    df = load_data(args.input)
    
    # Create visualization
    print("Generating comparison figure...")
    fig, axes = create_comparison_plot(df, args.output)
    
    # Print summary table
    print_summary_table(df)
    
    # Show plot if requested
    if args.show:
        plt.show()
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
