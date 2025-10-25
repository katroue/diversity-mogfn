#!/usr/bin/env python3
"""
Expand loss_ablation.yaml from grouped format to flat format.

The original config has a grouped structure:
    experiments:
      - group: "base_loss_comparison"
        vary: {base_loss: [TB, DB, ...]}
        
This script converts it to a flat structure expected by run_ablation_study.py:
    experiments:
      - name: "base_loss_comparison_TB"
        base_loss: "trajectory_balance"
      - name: "base_loss_comparison_DB"
        base_loss: "detailed_balance"
      ...

Usage:
    python scripts/expand_loss_ablation.py --input configs/ablations/loss_ablation.yaml --output configs/ablations/loss_ablation_final.yaml
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any


def expand_group_to_experiments(group: Dict[str, Any], 
                                ablation_factors: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a single experiment group into individual experiments.
    
    Args:
        group: Group configuration with 'group', 'fixed', 'vary' keys
        ablation_factors: Factor definitions to look up parameter details
    
    Returns:
        List of individual experiment configurations
    """
    group_name = group['group']
    fixed_params = group.get('fixed', {})
    vary_params = group.get('vary', {})
    
    experiments = []
    
    # Get factor being varied (usually just one per group)
    for factor_name, factor_values in vary_params.items():
        # Get factor definitions
        factor_options = ablation_factors.get(factor_name, {}).get('options', [])
        
        # Create experiment for each value
        for value in factor_values:
            # Find full parameter definition
            param_def = None
            for option in factor_options:
                if option.get('name') == value:
                    param_def = option
                    break
            
            # Create experiment config
            exp_name = f"{group_name}_{value}"
            
            exp = {
                'name': exp_name,
                'group': group_name,
                'description': f"{group.get('description', '')} - {value}",
                **fixed_params,  # Add fixed parameters from group
            }
            
            # Add the varied parameter
            if param_def:
                # Include full parameter definition
                exp[factor_name] = param_def.get('name')
                exp[f'{factor_name}_type'] = param_def.get('type')
                exp[f'{factor_name}_params'] = param_def.get('params', {})
                exp[f'{factor_name}_label'] = param_def.get('label', value)
            else:
                # Just use the value
                exp[factor_name] = value
            
            experiments.append(exp)
    
    return experiments


def expand_config(input_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Expand grouped loss ablation config to flat format.
    
    Args:
        input_path: Path to grouped config YAML
        output_path: Path to save flat config (optional)
    
    Returns:
        Expanded config dictionary
    """
    # Load config
    print(f"Loading config from {input_path}...")
    with open(input_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if already flat
    if 'experiments' in config and config['experiments']:
        first_exp = config['experiments'][0]
        if 'name' in first_exp and 'group' not in first_exp:
            print("⚠️  Config appears to already be in flat format!")
            return config
    
    # Get ablation factors for parameter lookups
    ablation_factors = config.get('ablation_factors', {})
    
    # Expand each group
    print("\nExpanding experiment groups...")
    all_experiments = []
    
    for i, group in enumerate(config.get('experiments', []), 1):
        group_name = group.get('group', f'group_{i}')
        print(f"\n  Group {i}: {group_name}")
        
        # Expand group to individual experiments
        group_experiments = expand_group_to_experiments(group, ablation_factors)
        
        print(f"    → Expanded to {len(group_experiments)} experiments")
        for exp in group_experiments:
            print(f"       • {exp['name']}")
        
        all_experiments.extend(group_experiments)
    
    # Update config with flat experiments list
    config['experiments'] = all_experiments
    
    # Also need to update the 'fixed' section for run_ablation_study.py
    if 'fixed_config' in config:
        # Map fixed_config to fixed (expected by run_ablation_study.py)
        config['fixed'] = config['fixed_config']
        
        # Extract seeds
        if 'num_seeds' in config['fixed']:
            num_seeds = config['fixed']['num_seeds']
            base_seed = config['fixed'].get('base_seed', 42)
            config['fixed']['seeds'] = [base_seed + i * 111 for i in range(num_seeds)]
    
    print(f"\n{'='*70}")
    print(f"Expansion complete!")
    print(f"{'='*70}")
    print(f"Original groups: {len(config.get('experiments', []))} groups")
    print(f"Expanded experiments: {len(all_experiments)} experiments")
    
    if 'fixed' in config and 'seeds' in config['fixed']:
        num_seeds = len(config['fixed']['seeds'])
        print(f"Seeds: {num_seeds}")
        print(f"Total runs: {len(all_experiments)} × {num_seeds} = {len(all_experiments) * num_seeds}")
    
    # Save if output path provided
    if output_path:
        print(f"\nSaving flat config to {output_path}...")
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved!")
    
    return config


def verify_expansion(config: Dict[str, Any]):
    """Verify the expanded config is valid."""
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    experiments = config.get('experiments', [])
    
    # Check all experiments have required fields
    required_fields = ['name', 'group']
    issues = []
    
    for i, exp in enumerate(experiments):
        for field in required_fields:
            if field not in exp:
                issues.append(f"Experiment {i} missing '{field}'")
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✓ All experiments have required fields")
    
    # Check for duplicates
    names = [exp['name'] for exp in experiments]
    if len(names) != len(set(names)):
        print("⚠️  Duplicate experiment names found!")
        from collections import Counter
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        for dup in duplicates:
            print(f"  • '{dup}' appears {Counter(names)[dup]} times")
    else:
        print(f"✓ All {len(names)} experiment names are unique")
    
    # Summary by group
    print("\nExperiments by group:")
    from collections import defaultdict
    by_group = defaultdict(int)
    for exp in experiments:
        by_group[exp.get('group', 'unknown')] += 1
    
    for group_name, count in sorted(by_group.items()):
        print(f"  • {group_name}: {count} experiments")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Expand grouped loss ablation config to flat format'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='loss_ablation.yaml',
        help='Input grouped config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='loss_ablation_FLAT.yaml',
        help='Output flat config file'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the expanded config'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LOSS ABLATION CONFIG EXPANDER")
    print("="*70 + "\n")
    
    # Expand config
    expanded_config = expand_config(args.input, args.output)
    
    # Verify if requested
    if args.verify:
        verify_expansion(expanded_config)
    
    print("✓ Done!\n")
    print("Next steps:")
    print(f"  1. Review the expanded config: {args.output}")
    print(f"  2. Run ablation study:")
    print(f"     python run_ablation_study.py \\")
    print(f"         --config {args.output} \\")
    print(f"         --ablation loss \\")
    print(f"         --output_dir results/ablations/loss")
    print()


if __name__ == '__main__':
    main()