"""
Baseline algorithms for multi-objective optimization comparison.

This module implements baselines for comparing against MOGFN-PC:
- RandomSampler: Random trajectory sampling baseline
- NSGA2Adapter: NSGA-II genetic algorithm using pymoo (optional)
- HN_GFN: Hypernetwork-GFlowNet (NeurIPS 2023)
- (Future) SingleObjectiveGFN: Train separate GFlowNet per objective
"""

from .random_sampler import RandomSampler
from .hn_gfn import HN_GFN

__all__ = ['RandomSampler', 'HN_GFN']

# Try to import NSGA2Adapter (requires pymoo)
try:
    from .nsga2_adapter import NSGA2Adapter
    __all__.append('NSGA2Adapter')
except ImportError:
    pass  # pymoo not available