"""
Baseline algorithms for multi-objective optimization comparison.

This module implements baselines for comparing against MOGFN-PC:
- RandomSampler: Random trajectory sampling baseline
- NSGA2Adapter: NSGA-II genetic algorithm using pymoo (optional)
- (Future) SingleObjectiveGFN: Train separate GFlowNet per objective
- (Future) HN_GFN: Hindsight GFlowNet
"""

from .random_sampler import RandomSampler

__all__ = ['RandomSampler']

# Try to import NSGA2Adapter (requires pymoo)
try:
    from .nsga2_adapter import NSGA2Adapter
    __all__.append('NSGA2Adapter')
except ImportError:
    pass  # pymoo not available