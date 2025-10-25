#!/usr/bin/env python3
"""Usage: 
        Capacity :
        python scripts/create_summary_report.py \
            --results_csv results/ablations/capacity/all_results.csv \
            --ablation capacity \
            --output_dir results/ablations/capacity/report
            
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
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

# Import the function from your ablation script
sys.path.append(str(Path(__file__).parent.parent))
# from scripts.run_ablation_study import create_summary_report

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
        output_dir=Path(args.output_dir)
    )

def create_summary_report(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Génère des rapports sommaires et CSV groupés.
    Écrit :
      - all_results.csv (raw)
      - summary_by_capacity.csv
      - summary_by_conditioning.csv
      - summary_by_capacity_and_conditioning.csv
      - overall_summary.csv (describe)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde brute
    results_df.to_csv(output_dir / 'all_results.csv', index=False)

    # S'assurer que les colonnes de groupement existent
    for col in ('capacity', 'conditioning'):
        if col not in results_df.columns:
            results_df[col] = np.nan

    # Sélection des colonnes numériques pour l'agrégation
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # Rien à agréger : on sauvegarde juste le dataframe complet et on sort
        results_df.to_csv(output_dir / 'all_results_no_numeric.csv', index=False)
        return

    def _agg_and_save(df, groupby_cols, out_name):
        grp = df.groupby(groupby_cols)[numeric_cols].agg(['mean', 'std', 'count'])
        # Aplatir les MultiIndex de colonnes
        grp.columns = ['_'.join(map(str, c)).strip() for c in grp.columns.values]
        grp = grp.reset_index()
        grp.to_csv(output_dir / out_name, index=False)

    # Groupements demandés
    _agg_and_save(results_df, 'capacity', 'summary_by_capacity.csv')
    _agg_and_save(results_df, 'conditioning', 'summary_by_conditioning.csv')
    _agg_and_save(results_df, ['capacity', 'conditioning'], 'summary_by_capacity_and_conditioning.csv')

    # Résumé global (describe) — convertir en DF et sauver
    overall = results_df.describe(include='all')
    try:
        overall_df = overall.transpose().reset_index().rename(columns={'index': 'metric'})
        overall_df.to_csv(output_dir / 'overall_summary.csv', index=False)
    except Exception:
        # Si describe renvoie des types non sérialisables, sauvegarder la version simple
        results_df[numeric_cols].describe().transpose().reset_index().to_csv(
            output_dir / 'overall_summary_numeric.csv', index=False
        )

if __name__ == '__main__':
    main()