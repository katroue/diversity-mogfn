## 1.5. Average Results Across Seeds

# This cell combines results from the 5 different seeds (42, 123, 456, 789, 1011)
# by computing the mean and standard deviation for each configuration

print("="*70)
print("AVERAGING RESULTS ACROSS SEEDS")
print("="*70)

# Check if 'seed' column exists
if 'seed' in df.columns:
    print(f"\n✓ Found seed column with values: {sorted(df['seed'].unique())}")
    print(f"  Total experiments before averaging: {len(df)}")

    # Identify grouping columns (all except seed and metrics)
    # Typically these are: exp_name, and any configuration parameters
    group_cols = []

    # Check for exp_name (primary grouping)
    if 'exp_name' in df.columns:
        # Remove seed suffix from exp_name if it exists
        df['config_name'] = df['exp_name'].str.replace(r'_seed\d+', '', regex=True)
        group_cols.append('config_name')
        print(f"  Unique configurations: {df['config_name'].nunique()}")

    # Add other configuration columns if they exist
    config_columns = ['temperature', 'sampling_strategy', 'on_policy',
                     'preference_sampling', 'alpha', 'batch_size',
                     'capacity', 'conditioning', 'loss', 'hidden_dim', 'num_layers']

    for col in config_columns:
        if col in df.columns and col not in group_cols:
            group_cols.append(col)

    if not group_cols:
        print("  ⚠ Warning: Could not identify grouping columns, using exp_name")
        group_cols = ['exp_name']

    print(f"  Grouping by: {group_cols}")

    # Separate into metrics and non-metrics
    metric_cols = [col for col in df.columns if col in available_metrics or
                   col in ['training_time', 'num_parameters', 'final_loss']]

    print(f"  Metrics to average: {len(metric_cols)}")

    # Group and compute mean/std
    df_mean = df.groupby(group_cols)[metric_cols].mean().reset_index()
    df_std = df.groupby(group_cols)[metric_cols].std().reset_index()

    # Add std columns with _std suffix
    for col in metric_cols:
        if col in df_std.columns:
            df_mean[f'{col}_std'] = df_std[col]

    # Add count of seeds used
    df_mean['num_seeds'] = df.groupby(group_cols).size().values

    # Store original data
    df_original = df.copy()

    # Replace df with averaged data
    df = df_mean.copy()

    # If we created config_name, use it as exp_name
    if 'config_name' in df.columns:
        df['exp_name'] = df['config_name']
        df = df.drop('config_name', axis=1)

    print(f"\n✓ Averaging complete!")
    print(f"  Configurations after averaging: {len(df)}")
    print(f"  Seeds per configuration: {df['num_seeds'].iloc[0] if 'num_seeds' in df.columns else 'N/A'}")
    print(f"  Columns with std: {len([c for c in df.columns if '_std' in c])}")

    # Display sample of averaged data
    print("\n📊 Sample of averaged data:")
    display_cols = ['exp_name'] + available_metrics[:5]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].head())

else:
    print("\n⚠ No 'seed' column found in data")
    print("  Proceeding with original data (assuming it's already averaged)")

print("\n" + "="*70)
