import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

# Load the results
df = pd.read_csv('results/ablations/sampling/all_results.csv')

# Load the YAML configuration to get experiment groupings
with open('configs/ablations/sampling_ablation.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define experiment categories based on YAML structure
experiment_categories = {
    'Temperature': ['temp_low', 'temp_medium', 'temp_high', 'temp_very_high'],
    'Strategy': ['greedy', 'categorical', 'top_k', 'top_p'],
    'Off-Policy': ['on_policy_pure', 'off_policy_10', 'off_policy_25', 'off_policy_50'],
    'Preference': ['pref_uniform', 'pref_dirichlet_low', 'pref_dirichlet_medium', 'pref_dirichlet_high'],
    'Batch Size': ['batch_32', 'batch_64', 'batch_256', 'batch_512'],
    'Combined': ['diverse_sampling', 'quality_sampling']
}

# Four most discriminative diversity metrics for sampling strategies
# Selected based on coefficient of variation (CV) across experiments:
# - pas: 86.5% CV (highest variation - most sensitive to sampling)
# - num_modes: 74.3% CV (very interpretable, high variation)
# - mce: 57.6% CV (mode coverage in objective space)
# - qds: 33.3% CV (quality-diversity balance)
metrics = {
    'mce': 'Mode Coverage Entropy',
    'pas': 'Preference Aligned Spread',
    'qds': 'Quality Diversity Score',
    'num_modes': 'Number of Modes'
}

# Extract experiment name from exp_name column (remove seed suffix)
df['experiment'] = df['exp_name'].str.rsplit('_seed', n=1).str[0]

# Calculate mean and std for each experiment across seeds
stats_df = df.groupby('experiment')[list(metrics.keys())].agg(['mean', 'std']).reset_index()

# Function to create grouped bar chart for a category
def plot_category(category_name, exp_names, figsize=(14, 8)):
    # Filter data for this category
    category_data = stats_df[stats_df['experiment'].isin(exp_names)]
    
    if len(category_data) == 0:
        print(f"Warning: No data found for category {category_name}")
        return
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Number of experiments and metrics
    n_experiments = len(category_data)
    n_metrics = len(metrics)
    
    # Set width of bars and positions
    bar_width = 0.2
    x = np.arange(n_experiments)
    
    # Colors for each metric
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Plot bars for each metric
    for i, (metric_key, metric_label) in enumerate(metrics.items()):
        means = category_data[(metric_key, 'mean')].values
        stds = category_data[(metric_key, 'std')].values
        
        offset = (i - n_metrics/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, 
                     label=metric_label, 
                     color=colors[i],
                     alpha=0.8,
                     yerr=stds,
                     capsize=3,
                     error_kw={'linewidth': 1, 'alpha': 0.6})
    
    # Customize the plot
    ax.set_xlabel('Sampling Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Diversity Metrics Comparison - {category_name}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    # Clean up experiment names for display
    display_names = [exp.replace('_', ' ').title() for exp in category_data['experiment'].values]
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path('results/ablations/sampling/report')
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'grouped_bar_{category_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / filename}")
    plt.close()

# Create a comprehensive comparison plot with all experiments
def plot_all_experiments(figsize=(20, 10)):
    # Get top performers by each metric
    top_experiments = set()
    for metric_key in metrics.keys():
        top_5 = stats_df.nlargest(5, (metric_key, 'mean'))['experiment'].values
        top_experiments.update(top_5)
    
    # Filter to top experiments
    top_data = stats_df[stats_df['experiment'].isin(top_experiments)].copy()
    top_data = top_data.sort_values((list(metrics.keys())[0], 'mean'), ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_experiments = len(top_data)
    n_metrics = len(metrics)
    bar_width = 0.2
    x = np.arange(n_experiments)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (metric_key, metric_label) in enumerate(metrics.items()):
        means = top_data[(metric_key, 'mean')].values
        stds = top_data[(metric_key, 'std')].values
        
        offset = (i - n_metrics/2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, 
               label=metric_label, 
               color=colors[i],
               alpha=0.8,
               yerr=stds,
               capsize=3,
               error_kw={'linewidth': 1, 'alpha': 0.6})
    
    ax.set_xlabel('Sampling Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=13, fontweight='bold')
    ax.set_title('Top Performing Sampling Strategies - All Diversity Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    display_names = [exp.replace('_', ' ').title() for exp in top_data['experiment'].values]
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_dir = Path('results/ablations/sampling/report')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'grouped_bar_top_performers.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'grouped_bar_top_performers.png'}")
    plt.close()

# Generate summary statistics
def print_summary_stats():
    print("\n" + "="*80)
    print("SAMPLING ABLATION SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    for metric_key, metric_label in metrics.items():
        print(f"\n{metric_label} ({metric_key}):")
        print("-" * 60)
        top_5 = stats_df.nlargest(5, (metric_key, 'mean'))
        for idx, row in top_5.iterrows():
            exp_name = str(row['experiment']).replace('_', ' ').title()
            mean_val = row[(metric_key, 'mean')]
            std_val = row[(metric_key, 'std')]
            print(f"  {exp_name:35s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n" + "="*80 + "\n")

# Main execution
if __name__ == "__main__":
    print("Generating grouped bar charts for sampling ablation study...")
    print(f"Total experiments found: {len(stats_df)}")
    
    # Plot each category
    for category_name, exp_names in experiment_categories.items():
        print(f"\nProcessing category: {category_name}")
        plot_category(category_name, exp_names)
    
    # Plot top performers across all experiments
    print("\nGenerating overall top performers plot...")
    plot_all_experiments()
    
    # Print summary statistics
    print_summary_stats()
    
    print("\nAll plots generated successfully!")
    print("Check: results/ablations/sampling/report")
