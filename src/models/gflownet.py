"""
Base GFlowNet utilities.

This module provides core data structures and interfaces for GFlowNet implementations.
MOGFN_PC (in mogfn_pc.py) is the main implementation used in this project.
"""

import torch
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Trajectory:
    """Container for GFlowNet trajectory data."""
    states: List[torch.Tensor]
    actions: List[int]
    log_probs: List[torch.Tensor]
    is_terminal: bool
    reward: Optional[float] = None

    def __len__(self):
        return len(self.actions)


class GFlowNetEnvironment(ABC):
    """
    Abstract base class for GFlowNet environments.

    This interface is implemented by:
    - HyperGrid (src/environments/hypergrid.py)
    - DNASequences (src/environments/sequences.py)
    - Molecules (src/environments/molecules.py)
    - NGrams (src/environments/ngrams.py)
    """

    @abstractmethod
    def get_initial_state(self) -> torch.Tensor:
        """Get initial state."""
        pass

    @abstractmethod
    def step(self, state: torch.Tensor, action: int):
        """
        Take action in environment.

        Returns:
            next_state: Next state after action
            reward: Reward (multi-objective vector or scalar)
            done: Whether episode is terminal
            info: Additional information dict
        """
        pass

    @abstractmethod
    def get_valid_actions(self, state: torch.Tensor) -> List[int]:
        """Get list of valid actions for state."""
        pass

    @abstractmethod
    def compute_reward(self, state: torch.Tensor):
        """Compute reward for terminal state (multi-objective)."""
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of state representation."""
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Number of possible actions."""
        pass



