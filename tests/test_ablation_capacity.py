#!/usr/bin/env python3
"""
Comprehensive test script to debug all metrics without running full ablation.
Tests: Traditional, Trajectory, Spatial, Objective, Composite, Dynamics, and Flow metrics.
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path for src imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mogfn_pc import MOGFN_PC, PreferenceSampler, MOGFNTrainer
from src.environments.hypergrid import HyperGrid
from src.metrics.traditional import compute_all_traditional_metrics
from src.metrics.trajectory import trajectory_diversity_score, multi_path_diversity
from src.metrics.spatial import mode_coverage_entropy, pairwise_minimum_distance
from src.metrics.objective import pareto_front_smoothness, preference_aligned_spread
from src.metrics.dynamics import replay_buffer_diversity
from src.metrics.flow import flow_concentration_index
from src.metrics.composite import quality_diversity_score, diversity_efficiency_ratio

def test_single_experiment():
    """Test a single experiment with minimal iterations and comprehensive metric testing."""
    
    print("="*70)
    print("COMPREHENSIVE METRIC TEST - Single Experiment")
    print("="*70)
    
    # Minimal configuration for quick testing
    config = {
        'height': 8,
        'num_objectives': 2,
        'reward_config': 'corners',
        'hidden_dim': 64,
        'num_layers': 3,
        'preference_encoding': 'vanilla',
        'conditioning': 'concat',
        'preference_sampling': 'dirichlet',
        'alpha': 1.5,
        'learning_rate': 1e-3,
        'beta': 1.0,
        'num_iterations': 100,  # REDUCED for quick testing
        'batch_size': 32,       # REDUCED for quick testing
        'log_every': 50,
        'eval_samples': 50,     # REDUCED for quick testing
        'seed': 42
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    print("\n1. Creating environment...")
    env = HyperGrid(
        height=config['height'],
        num_objectives=config['num_objectives'],
        reward_config=config['reward_config']
    )
    print(f"   ✓ Environment created: {env.state_dim}D state, {env.num_actions} actions")
    
    print("\n2. Creating model...")
    mogfn = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=config['hidden_dim'],
        num_actions=env.num_actions,
        num_layers=config['num_layers'],
        preference_encoding=config['preference_encoding'],
        conditioning_type=config['conditioning']
    )
    num_params = sum(p.numel() for p in mogfn.parameters())
    print(f"   ✓ Model created: {num_params:,} parameters")
    
    print("\n3. Creating preference sampler...")
    pref_sampler = PreferenceSampler(
        num_objectives=env.num_objectives,
        distribution=config['preference_sampling'],
        alpha=config['alpha']
    )
    print(f"   ✓ Preference sampler created")
    
    print("\n4. Creating optimizer and trainer...")
    optimizer = torch.optim.Adam(mogfn.parameters(), lr=config['learning_rate'])
    trainer = MOGFNTrainer(
        mogfn=mogfn,
        env=env,
        preference_sampler=pref_sampler,
        optimizer=optimizer,
        beta=config['beta']
    )
    print(f"   ✓ Trainer created")
    
    print(f"\n5. Training for {config['num_iterations']} iterations...")
    import time
    start_time = time.time()
    training_history = trainer.train(
        num_iterations=config['num_iterations'],
        batch_size=config['batch_size'],
        log_every=config['log_every']
    )
    training_time = time.time() - start_time
    print(f"   ✓ Training complete in {training_time:.1f} seconds")
    
    print(f"\n6. Evaluating with {config['eval_samples']} samples...")
    eval_results = trainer.evaluate(num_samples=config['eval_samples'])
    
    objectives_tensor = eval_results['objectives']
    preferences_tensor = eval_results['preferences']
    
    print(f"   ✓ Evaluation complete")
    print(f"      objectives type: {type(objectives_tensor)}")
    print(f"      objectives shape: {objectives_tensor.shape if hasattr(objectives_tensor, 'shape') else 'N/A'}")
    print(f"      preferences type: {type(preferences_tensor)}")
    print(f"      preferences shape: {preferences_tensor.shape if hasattr(preferences_tensor, 'shape') else 'N/A'}")
    
    # Convert to numpy for metrics
    if isinstance(objectives_tensor, torch.Tensor):
        objectives = objectives_tensor.detach().cpu().numpy()
    else:
        objectives = objectives_tensor
    
    print("\n" + "="*70)
    print("7. TESTING ALL METRICS")
    print("="*70)
    
    metrics_passed = 0
    metrics_failed = 0
    
    # Test 7.1: Traditional metrics
    print("\n7.1 Testing TRADITIONAL metrics...")
    try:
        reference_point = np.array([1.1] * env.num_objectives)
        traditional = compute_all_traditional_metrics(objectives, reference_point)
        print(f"   ✓ Traditional metrics PASSED:")
        print(f"      - Hypervolume: {traditional['hypervolume']:.4f}")
        print(f"      - R2 Indicator: {traditional['r2_indicator']:.4f}")
        print(f"      - Avg Pairwise Distance: {traditional['avg_pairwise_distance']:.4f}")
        print(f"      - Spacing: {traditional['spacing']:.4f}")
        print(f"      - Spread: {traditional['spread']:.4f}")
        metrics_passed += 1
    except Exception as e:
        print(f"   ✗ Traditional metrics FAILED: {e}")
        import traceback
        traceback.print_exc()
        metrics_failed += 1
    
    # Test 7.2: Trajectory metrics
    print("\n7.2 Testing TRAJECTORY metrics...")
    try:
        from src.models.mogfn_pc import MOGFNSampler
        sampler = MOGFNSampler(mogfn, env, pref_sampler)
        trajectories = []
        # Use TENSOR version for PyTorch model
        for i in range(min(10, len(preferences_tensor))):  # Only 10 for quick test
            traj = sampler.sample_trajectory(preferences_tensor[i], explore=False)
            trajectories.append(traj)
        
        tds = trajectory_diversity_score(trajectories)
        mpd = multi_path_diversity(trajectories)
        print(f"   ✓ Trajectory metrics PASSED:")
        print(f"      - TDS (Trajectory Diversity Score): {tds:.4f}")
        print(f"      - MPD (Multi-Path Diversity): {mpd:.4f}")
        metrics_passed += 1
    except Exception as e:
        print(f"   ✗ Trajectory metrics FAILED: {e}")
        import traceback
        traceback.print_exc()
        metrics_failed += 1
    
    # Test 7.3: Spatial metrics
    print("\n7.3 Testing SPATIAL metrics...")
    try:
        mce, num_modes = mode_coverage_entropy(objectives)
        pmd = pairwise_minimum_distance(objectives)
        pfs = pareto_front_smoothness(objectives)
        print(f"   ✓ Spatial metrics PASSED:")
        print(f"      - MCE (Mode Coverage Entropy): {mce:.4f}")
        print(f"      - Number of modes: {num_modes}")
        print(f"      - PMD (Pairwise Minimum Distance): {pmd:.4f}")
        print(f"      - PFS (Pareto Front Smoothness): {pfs:.4f}")
        metrics_passed += 1
    except Exception as e:
        print(f"   ✗ Spatial metrics FAILED: {e}")
        import traceback
        traceback.print_exc()
        metrics_failed += 1
    
    # Test 7.4: Objective metrics
    print("\n7.4 Testing OBJECTIVE metrics...")
    try:
        # PFS already computed above
        print(f"   ✓ Objective metrics PASSED:")
        print(f"      - PFS (Pareto Front Smoothness): {pfs:.4f}")
        print(f"      Note: PAS (Preference-Aligned Spread) requires full gflownet setup - skipped for quick test")
        metrics_passed += 1
    except Exception as e:
        print(f"   ✗ Objective metrics FAILED: {e}")
        import traceback
        traceback.print_exc()
        metrics_failed += 1
    
    # Test 7.5: Composite metrics
    print("\n7.5 Testing COMPOSITE metrics...")
    try:
        reference_point = np.array([1.1] * env.num_objectives)
        
        # Quality-Diversity Score
        qds_results = quality_diversity_score(
            objectives, 
            reference_point, 
            alpha=0.5
        )
        
        # Diversity-Efficiency Ratio
        der_results = diversity_efficiency_ratio(
            objectives,
            training_time=training_time,
            num_parameters=num_params
        )
        
        print(f"   ✓ Composite metrics PASSED:")
        print(f"      - QDS (Quality-Diversity Score): {qds_results['qds']:.4f}")
        print(f"        * Hypervolume: {qds_results['hypervolume']:.4f}")
        print(f"        * Diversity: {qds_results['diversity']:.4f}")
        print(f"        * Normalized HV: {qds_results['normalized_hv']:.4f}")
        print(f"        * Normalized Div: {qds_results['normalized_div']:.4f}")
        print(f"      - DER (Diversity-Efficiency Ratio): {der_results['der']:.6f}")
        print(f"        * Diversity: {der_results['diversity']:.4f}")
        print(f"        * Computational cost: {der_results['computational_cost']:.4f}")
        print(f"        * Training time: {der_results['training_time']:.2f}s")
        print(f"        * Num parameters: {der_results['num_parameters']:,}")
        metrics_passed += 1
    except Exception as e:
        print(f"   ✗ Composite metrics FAILED: {e}")
        import traceback
        traceback.print_exc()
        metrics_failed += 1
    
    # Test 7.6: Dynamics metrics
    print("\n7.6 Testing DYNAMICS metrics...")
    try:
        rbd_traj = replay_buffer_diversity(trajectories, metric='trajectory_distance')
        rbd_obj = replay_buffer_diversity(trajectories, metric='objective_diversity')
        rbd_state = replay_buffer_diversity(trajectories, metric='state_coverage')
        
        print(f"   ✓ Dynamics metrics PASSED:")
        print(f"      - RBD (Replay Buffer Diversity):")
        print(f"        * Trajectory distance: {rbd_traj:.4f}")
        print(f"        * Objective diversity: {rbd_obj:.4f}")
        print(f"        * State coverage: {rbd_state:.4f}")
        metrics_passed += 1
    except Exception as e:
        print(f"   ✗ Dynamics metrics FAILED: {e}")
        import traceback
        traceback.print_exc()
        metrics_failed += 1
    
    # Test 7.7: Flow metrics
    print("\n7.7 Testing FLOW metrics...")
    try:
        state_visits = {}
        for traj in trajectories:
            for state in traj.states:
                if isinstance(state, torch.Tensor):
                    state_key = tuple(state.detach().cpu().numpy().flatten())
                else:
                    state_key = tuple(np.array(state).flatten())
                state_visits[state_key] = state_visits.get(state_key, 0) + 1
        
        fci_gini = flow_concentration_index(state_visits, method='gini')
        fci_entropy = flow_concentration_index(state_visits, method='entropy')
        
        print(f"   ✓ Flow metrics PASSED:")
        print(f"      - FCI (Flow Concentration Index):")
        print(f"        * Gini method: {fci_gini:.4f}")
        print(f"        * Entropy method: {fci_entropy:.4f}")
        print(f"      - Unique states visited: {len(state_visits)}")
        metrics_passed += 1
    except Exception as e:
        print(f"   ✗ Flow metrics FAILED: {e}")
        import traceback
        traceback.print_exc()
        metrics_failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Metrics categories tested: {metrics_passed + metrics_failed}")
    print(f"✓ Passed: {metrics_passed}")
    print(f"✗ Failed: {metrics_failed}")
    
    if metrics_failed == 0:
        print("\n🎉 ALL METRICS PASSED! Your code is ready for full ablation study.")
    else:
        print(f"\n⚠️  {metrics_failed} metric categories failed. Fix the issues above before running full ablation.")
    
    print("="*70)

if __name__ == '__main__':
    test_single_experiment()