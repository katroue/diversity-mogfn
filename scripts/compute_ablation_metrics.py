from src.metrics.traditional import compute_all_traditional_metrics
from src.metrics.trajectory import trajectory_diversity_score, multi_path_diversity
from src.metrics.spatial import mode_coverage_entropy, pairwise_minimum_distance
from src.metrics.objective import preference_aligned_spread
from src.metrics.dynamics import replay_buffer_diversity
from src.metrics.flow import flow_concentration_index
from src.metrics.composite import quality_diversity_score, diversity_efficiency_ratio
import pandas as pd

results = []

for experiment in experiments:
    # Load results
    objectives = load_objectives(experiment)
    trajectories = load_trajectories(experiment)
    
    # Compute all metrics
    metrics = {}
    metrics.update(compute_all_traditional_metrics(objectives, ref_point))
    metrics['tds'] = trajectory_diversity_score(trajectories)
    metrics['mpd'] = multi_path_diversity(trajectories)
    metrics['mce'], _ = mode_coverage_entropy(objectives)
    metrics['pmd'] = pairwise_minimum_distance(objectives)
    metrics['pas'] = preference_aligned_spread(objectives, experiment.preference_directions)
    metrics['rbd'] = replay_buffer_diversity(experiment.replay_buffer)
    metrics['fci'] = flow_concentration_index(experiment.flow_data)
    metrics['qds'] = quality_diversity_score(objectives, trajectories)
    metrics['der'] = diversity_efficiency_ratio(objectives, trajectories)
    
    # Add metadata
    metrics['capacity'] = experiment.capacity
    metrics['arch_type'] = experiment.arch_type
    metrics['num_params'] = count_parameters(model)
    metrics['training_time'] = experiment.training_time
    
    results.append(metrics)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('results/ablations/capacity/all_metrics.csv')