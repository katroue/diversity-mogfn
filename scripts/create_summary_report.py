#!/usr/bin/env python3
"""Usage: 
        Capacity :
        python scripts/create_summary_report.py \
            --results_csv results/ablations/capacity/all_results.csv \
            --ablation capacity \
            --output_dir results/ablations/capacity
            
        Sampling :
        python scripts/create_summary_report.py \
            --results_csv results/ablations/sampling/all_results.csv \
            --ablation sampling \
            --output_dir results/ablations/sampling
        
        Loss :
        python scripts/create_summary_report.py \
            --results_csv results/ablations/loss/all_results.csv \
            --ablation loss \
            --output_dir results/ablations/loss
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

# Import the function from your ablation script
sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_ablation_study import create_summary_report

def main():
    parser = argparse.ArgumentParser(
        description='Create summary report from ablation results CSV'
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
        default='results/ablations',
        help='Output directory for report files'
    )
    args = parser.parse_args()

    results_df = pd.read_csv(args.results_csv)
    print("Creating summary report...")
    create_summary_report(
        results_df=results_df,
        ablation_type=args.ablation,
        output_dir=Path(args.output_dir)
    )

if __name__ == '__main__':
    main()