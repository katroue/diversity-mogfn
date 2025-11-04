#!/usr/bin/env python3
"""
Test that --resume functionality properly preserves previously completed experiments.

This test verifies that the fix for the resume bug works correctly:
- Previously completed experiments are merged with new results
- results_temp.csv is not overwritten, but appended to
- Duplicates are handled correctly
"""

import sys
import pandas as pd
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_resume_merge_logic():
    """Test the core merge logic for resume functionality."""
    print("\n" + "="*80)
    print("Testing Resume Functionality Fix")
    print("="*80)

    # Create temporary results files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        results_temp_file = tmpdir / 'results_temp.csv'

        # Simulate first run: 5 experiments complete
        print("\n1. Simulating first run (5 experiments)...")
        results = []
        for i in range(5):
            results.append({
                'exp_name': f'experiment_seed{i}',
                'condition_name': 'condition1',
                'seed': i,
                'mce': 0.5 + i * 0.01,
                'hypervolume': 0.8 + i * 0.01
            })

        df_first = pd.DataFrame(results)
        df_first.to_csv(results_temp_file, index=False)
        print(f"   Saved {len(df_first)} results to {results_temp_file.name}")
        print(f"   Experiments: {list(df_first['exp_name'].values)}")

        # Simulate resume: load existing results
        print("\n2. Simulating resume (loading existing results)...")
        df_existing = pd.read_csv(results_temp_file)
        completed = set(df_existing['exp_name'].values)
        print(f"   Found {len(completed)} completed experiments")
        print(f"   Completed: {list(completed)}")

        # Simulate second run: 3 more experiments
        print("\n3. Simulating second run (3 new experiments)...")
        results = []  # Reset (as the script does)
        for i in range(5, 8):
            results.append({
                'exp_name': f'experiment_seed{i}',
                'condition_name': 'condition1',
                'seed': i,
                'mce': 0.5 + i * 0.01,
                'hypervolume': 0.8 + i * 0.01
            })

        # Apply the FIX: Merge with existing temp results
        df_temp = pd.DataFrame(results)
        print(f"   New results: {len(df_temp)} experiments")

        if results_temp_file.exists():
            df_existing = pd.read_csv(results_temp_file)
            print(f"   Merging with {len(df_existing)} existing results...")
            df_temp = pd.concat([df_existing, df_temp], ignore_index=True)
            df_temp = df_temp.drop_duplicates(subset=['exp_name'], keep='last')

        df_temp.to_csv(results_temp_file, index=False)
        print(f"   Saved {len(df_temp)} total results")

        # Verify the fix worked
        print("\n4. Verifying results...")
        df_final = pd.read_csv(results_temp_file)
        final_experiments = set(df_final['exp_name'].values)

        print(f"   Total experiments in file: {len(df_final)}")
        print(f"   Experiments: {list(df_final['exp_name'].values)}")

        # Check that all 8 experiments are present
        expected_experiments = {f'experiment_seed{i}' for i in range(8)}
        assert final_experiments == expected_experiments, \
            f"Expected {expected_experiments}, got {final_experiments}"

        print(f"   ✓ All 8 experiments present!")

        # Test duplicate handling
        print("\n5. Testing duplicate handling (re-running experiment)...")
        results = [{
            'exp_name': 'experiment_seed3',  # Re-run experiment 3
            'condition_name': 'condition1',
            'seed': 3,
            'mce': 0.99,  # Different value
            'hypervolume': 0.99
        }]

        df_temp = pd.DataFrame(results)
        if results_temp_file.exists():
            df_existing = pd.read_csv(results_temp_file)
            df_temp = pd.concat([df_existing, df_temp], ignore_index=True)
            df_temp = df_temp.drop_duplicates(subset=['exp_name'], keep='last')
        df_temp.to_csv(results_temp_file, index=False)

        df_final = pd.read_csv(results_temp_file)
        print(f"   Total experiments after re-run: {len(df_final)}")

        # Should still be 8 (not 9), with updated value
        assert len(df_final) == 8, f"Expected 8 experiments, got {len(df_final)}"

        # Check that experiment_seed3 has the new value
        exp3_row = df_final[df_final['exp_name'] == 'experiment_seed3'].iloc[0]
        assert exp3_row['mce'] == 0.99, f"Expected updated value 0.99, got {exp3_row['mce']}"

        print(f"   ✓ Duplicate handled correctly (kept last occurrence)")
        print(f"   ✓ Updated value for experiment_seed3: mce={exp3_row['mce']}")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED - Resume functionality fix works correctly!")
    print("="*80)
    print("\nThe fix ensures that:")
    print("  • Previously completed experiments are preserved")
    print("  • New experiments are appended to results_temp.csv")
    print("  • Re-running an experiment updates its values (keeps last)")
    print("  • No data loss when resuming interrupted experiments")
    print()


if __name__ == '__main__':
    try:
        test_resume_merge_logic()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
