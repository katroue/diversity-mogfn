import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""
Analyse and plot interaction effects from factorial experiments on hypergrid environment.

Usage :
    Capacity x Temperature interaction plot:
    python scripts/factorials/hypergrid/analyse_hypergrid_factorial.py
        --input results/factorials/capacity_sampling/results_temp.csv
        --output results/factorials/capacity_sampling/interaction_plot.pdf
    
    Capacity x Loss interaction plot:
    python scripts/factorials/hypergrid/analyse_hypergrid_factorial.py
        --input results/factorials/capacity_loss/results_capacity_loss.csv
        --output results/factorials/capacity_loss/interaction_plot.pdf
    
    Sampling x Loss interaction plot:
    python scripts/factorials/hypergrid/analyse_hypergrid_factorial.py
        --input results/factorials/sampling_loss/results_sampling_loss.csv
        --output results/factorials/sampling_loss/interaction_plot.pdf

"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def create_plot(df, args):
    """
    Create interaction plot for capacity x temperature effects on MCE.
    
    Args:
        df: DataFrame with columns 'capacity_level', 'temperature_level', 'mce'
        args: Command-line arguments with output path
    """

    # Create interaction plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for temp in df['temperature_level'].unique():
        subset = df[df['temperature_level'] == temp]
        means = subset.groupby('capacity_level')['mce'].mean()
        ax.plot(means.index, means.values, 'o-', label=f'temp={temp}', linewidth=2)

    ax.set_xlabel('Capacity')
    ax.set_ylabel('Mode Coverage Entropy (MCE)')
    ax.set_title('Capacity × Temperature Interaction')
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output if args.output else args.input.parent / 'interaction_plot.pdf')

def main():
    parser = argparse.ArgumentParser(
        description='Run factorial experiments for Multi-Objective GFlowNets on N-grams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file for results (default: inferred from input path)'
    )
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)

    create_plot(df, args)

if __name__ == '__main__':
    main()