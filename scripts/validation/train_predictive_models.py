#!/usr/bin/env python3
"""
Train predictive models for diversity metrics from configuration parameters.

This script implements Validation 2: Predictive Power from Phase 4: Metric Validation.

It builds regression models to predict final diversity metrics from CONFIGURATION ONLY:
- Model capacity (small, medium, large, xlarge)
- Sampling temperature (low, high, very_high)
- Loss function (TB, DB, FM, SubTB variants)
- Architecture parameters (hidden_dim, num_layers)

This answers: "Can we predict diversity outcomes from experimental design choices?"

Usage:
    # Train models on ablation studies
    python scripts/validation/train_predictive_models.py --dataset ablations

    # Train models on factorial experiments
    python scripts/validation/train_predictive_models.py --dataset factorials

    # Train on both
    python scripts/validation/train_predictive_models.py --dataset all
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Target metrics to predict (final diversity metrics)
TARGET_METRICS = ['qds', 'mce', 'num_unique_solutions']

# Predictor categories
CONFIG_FEATURES = ['capacity', 'temperature', 'loss_type', 'hidden_dim', 'num_layers']
METRIC_FEATURES = ['hypervolume', 'tds', 'pfs', 'avg_pairwise_distance']


def load_data(dataset: str) -> pd.DataFrame:
    """Load experiment data."""
    if dataset == 'ablations':
        from compute_metric_correlations import load_ablation_data
        return load_ablation_data()
    elif dataset == 'factorials':
        from compute_metric_correlations import load_factorial_data
        return load_factorial_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix for prediction.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (feature_df, feature_names)
    """
    features = []
    feature_names = []

    # Configuration features (categorical)
    for col in CONFIG_FEATURES:
        if col in df.columns:
            # Encode categorical variables
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].fillna('unknown'))
                features.append(encoded)
                feature_names.append(f'{col}_encoded')
            else:
                features.append(df[col].fillna(df[col].median()).values)
                feature_names.append(col)

    # Metric features (numerical, as proxies for early metrics)
    for col in METRIC_FEATURES:
        if col in df.columns:
            # Use median imputation for missing values
            values = df[col].fillna(df[col].median()).values
            features.append(values)
            feature_names.append(col)

    # Stack features
    if features:
        X = np.column_stack(features)
        return pd.DataFrame(X, columns=feature_names), feature_names
    else:
        raise ValueError("No features available!")


def train_models(X: np.ndarray, y: np.ndarray, model_type: str = 'all') -> Dict:
    """
    Train regression models.

    Args:
        X: Feature matrix
        y: Target variable
        model_type: Type of model ('linear', 'ridge', 'rf', or 'all')

    Returns:
        Dictionary of trained models
    """
    models = {}

    if model_type in ['linear', 'all']:
        models['Linear Regression'] = LinearRegression()

    if model_type in ['ridge', 'all']:
        models['Ridge Regression'] = Ridge(alpha=1.0)

    if model_type in ['rf', 'all']:
        models['Random Forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    # Train each model
    for name, model in models.items():
        model.fit(X, y)

    return models


def evaluate_model_cv(X: np.ndarray, y: np.ndarray, model, cv_folds: int = 5) -> Dict:
    """
    Evaluate model using cross-validation.

    Args:
        X: Feature matrix
        y: Target variable
        model: Regression model
        cv_folds: Number of CV folds

    Returns:
        Dictionary with evaluation metrics
    """
    # Perform cross-validation
    r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    mae_scores = cross_val_score(model, X, y, cv=cv_folds,
                                 scoring='neg_mean_absolute_error')
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv_folds,
                                           scoring='neg_mean_squared_error'))

    return {
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
        'mae_mean': -mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std()
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                    target_name: str, output_path: Path):
    """
    Plot predicted vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        target_name: Name of target metric
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
           label='Perfect Prediction')

    # Compute R²
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Add metrics text
    text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(f'Actual {target_name.upper()}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted {target_name.upper()}', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Predicting {target_name.upper()}',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names: List[str], target_name: str,
                           output_path: Path):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        target_name: Target metric name
        output_path: Path to save plot
    """
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))

    # Plot horizontal bars
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importances[indices], alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance - Predicting {target_name.upper()}',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_model_performance(results: Dict[str, Dict], target_name: str):
    """
    Print model performance table.

    Args:
        results: Dictionary of model results
        target_name: Target metric name
    """
    print(f"\n{'='*80}")
    print(f"MODEL PERFORMANCE - Predicting {target_name.upper()}")
    print(f"{'='*80}\n")

    print(f"{'Model':<25s} {'R²':>10s} {'MAE':>10s} {'RMSE':>10s}")
    print("-" * 80)

    for model_name, metrics in results.items():
        r2 = metrics['r2_mean']
        mae = metrics['mae_mean']
        rmse = metrics['rmse_mean']

        # Color code R² performance
        if r2 > 0.8:
            status = "✓✓"
        elif r2 > 0.6:
            status = "✓"
        else:
            status = "⚠️"

        print(f"{model_name:<25s} {r2:>10.3f} {mae:>10.3f} {rmse:>10.3f}  {status}")

    print("\nInterpretation:")
    print("  R² > 0.8 : Excellent predictive power ✓✓")
    print("  R² > 0.6 : Good predictive power ✓")
    print("  R² > 0.4 : Moderate predictive power")
    print("  R² ≤ 0.4 : Poor predictive power ⚠️")


def save_results(results: Dict, feature_names: List[str], output_dir: Path):
    """
    Save prediction results to CSV.

    Args:
        results: Dictionary of results for all targets and models
        feature_names: List of feature names
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary table
    rows = []
    for target, models in results.items():
        for model_name, metrics in models.items():
            rows.append({
                'target_metric': target,
                'model': model_name,
                'r2_mean': metrics['r2_mean'],
                'r2_std': metrics['r2_std'],
                'mae_mean': metrics['mae_mean'],
                'mae_std': metrics['mae_std'],
                'rmse_mean': metrics['rmse_mean'],
                'rmse_std': metrics['rmse_std']
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / 'model_performance.csv', index=False)

    # Save feature list
    with open(output_dir / 'features_used.txt', 'w') as f:
        f.write("Features used for prediction:\n")
        f.write("="*50 + "\n\n")
        for i, feature in enumerate(feature_names, 1):
            f.write(f"{i}. {feature}\n")

    print(f"\n✓ Saved results to: {output_dir}")
    print(f"  - model_performance.csv")
    print(f"  - features_used.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Train predictive models for diversity metrics',
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

    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    args = parser.parse_args()

    print("="*80)
    print("PREDICTIVE MODEL TRAINING")
    print("Validation 2: Phase 4 - Metric Validation")
    print("="*80)
    print()

    # Create output directory
    output_base = Path('results/validation/predictive_models')
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
            print(f"✓ Loaded {len(df)} experiments")

            # Prepare features
            X_df, feature_names = prepare_features(df)
            print(f"✓ Prepared {len(feature_names)} features: {', '.join(feature_names)}")

            # Create output directory for this dataset
            output_dir = output_base / dataset_name
            output_dir.mkdir(exist_ok=True)

            # Train models for each target metric
            all_results = {}

            for target_metric in TARGET_METRICS:
                if target_metric not in df.columns:
                    print(f"\n⚠️  Target metric '{target_metric}' not found - skipping")
                    continue

                # Get target variable (drop NaN)
                valid_idx = df[target_metric].notna() & X_df.notna().all(axis=1)
                X = X_df[valid_idx].values
                y = df.loc[valid_idx, target_metric].values

                if len(y) < 10:
                    print(f"\n⚠️  Insufficient data for '{target_metric}' ({len(y)} samples) - skipping")
                    continue

                print(f"\n{'='*80}")
                print(f"TARGET: {target_metric.upper()}")
                print(f"{'='*80}")
                print(f"Training data: {len(y)} samples")

                # Train models
                models = train_models(X, y, model_type='all')
                print(f"✓ Trained {len(models)} models")

                # Evaluate each model
                model_results = {}
                for model_name, model in models.items():
                    cv_results = evaluate_model_cv(X, y, model, cv_folds=args.cv_folds)
                    model_results[model_name] = cv_results

                    # Plot predictions (train on all data for visualization)
                    y_pred = model.predict(X)
                    plot_path = output_dir / f'{target_metric}_{model_name.replace(" ", "_").lower()}_predictions.pdf'
                    plot_predictions(y, y_pred, model_name, target_metric, plot_path)
                    print(f"  ✓ Saved: {plot_path.name}")

                    # Plot feature importance for Random Forest
                    if model_name == 'Random Forest':
                        importance_path = output_dir / f'{target_metric}_feature_importance.pdf'
                        plot_feature_importance(model, feature_names, target_metric, importance_path)
                        print(f"  ✓ Saved: {importance_path.name}")

                all_results[target_metric] = model_results

                # Print performance table
                print_model_performance(model_results, target_metric)

            # Save all results
            if all_results:
                save_results(all_results, feature_names, output_dir)

        except Exception as e:
            print(f"\n✗ Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("PREDICTIVE MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_base}")
    print("\nKey Findings:")
    print("  - Check model_performance.csv for R² scores")
    print("  - Review prediction plots to assess model quality")
    print("  - Examine feature importance to understand key predictors")
    print("\nSuccess Criteria:")
    print("  ✓ R² > 0.6 indicates good predictive power")
    print("  ✓ Configuration features should be important predictors")


if __name__ == '__main__':
    main()
