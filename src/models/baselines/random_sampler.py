"""
Random Sampling Baseline for Multi-Objective Optimization.

Provides a simple baseline that samples random trajectories from the environment
without any learning or optimization. Useful for validating that learned methods
outperform random search.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RandomTrajectory:
    """Store a randomly sampled trajectory."""
    states: List[Any]
    actions: List[int]
    objectives: np.ndarray  # Shape: (num_objectives,)
    is_terminal: bool


class RandomSampler:
    """
    Random sampling baseline for multi-objective optimization.

    Samples random trajectories from the environment by taking uniformly
    random valid actions at each step until reaching a terminal state.

    Args:
        env: MultiObjectiveEnvironment instance
        max_steps: Maximum steps per trajectory (default: 100)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        env,
        max_steps: int = 100,
        seed: Optional[int] = None
    ):
        self.env = env
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)

        # Storage for sampled trajectories
        self.trajectories: List[RandomTrajectory] = []
        self.objectives_history: List[np.ndarray] = []

        logger.info(f"Initialized RandomSampler with max_steps={max_steps}, seed={seed}")

    def sample_trajectory(self) -> RandomTrajectory:
        """
        Sample a single random trajectory.

        Returns:
            RandomTrajectory with states, actions, and final objectives
        """
        state = self.env.get_initial_state()
        states = [state]
        actions = []
        is_done = False

        for step in range(self.max_steps):
            # Get valid actions
            valid_actions = self.env.get_valid_actions(state)
            if len(valid_actions) == 0:
                logger.warning(f"No valid actions at step {step}")
                break

            # Sample random action
            action = self.rng.choice(valid_actions)
            actions.append(action)

            # Take action
            next_state, done = self.env.step(state, action)
            state = next_state
            states.append(state)
            is_done = done

            if done:
                break

        # Get final objectives (convert to numpy)
        import torch
        objectives_tensor = self.env.compute_objectives(state)
        objectives = objectives_tensor.detach().cpu().numpy() if torch.is_tensor(objectives_tensor) else np.array(objectives_tensor)

        trajectory = RandomTrajectory(
            states=states,
            actions=actions,
            objectives=objectives,
            is_terminal=is_done
        )

        return trajectory

    def train(
        self,
        num_iterations: int,
        batch_size: int = 32,
        log_interval: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        'Train' by sampling random trajectories.

        Args:
            num_iterations: Number of iterations (each samples batch_size trajectories)
            batch_size: Number of trajectories per iteration
            log_interval: Log progress every N iterations
            **kwargs: Ignored (for API compatibility)

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting random sampling: {num_iterations} iterations × {batch_size} samples")

        total_sampled = 0
        history = {
            'num_sampled': [],
            'num_unique_solutions': [],
            'mean_objectives': []
        }

        for iteration in range(num_iterations):
            # Sample batch of trajectories
            batch_trajectories = []
            batch_objectives = []

            for _ in range(batch_size):
                traj = self.sample_trajectory()
                batch_trajectories.append(traj)
                batch_objectives.append(traj.objectives)

                # Only store terminal trajectories
                if traj.is_terminal:
                    self.trajectories.append(traj)
                    self.objectives_history.append(traj.objectives)

            total_sampled += batch_size

            # Log progress
            if (iteration + 1) % log_interval == 0 or iteration == 0:
                num_terminal = len(self.objectives_history)
                if num_terminal > 0:
                    unique_solutions = len(set(tuple(obj) for obj in self.objectives_history))
                    mean_obj = np.mean(self.objectives_history, axis=0)

                    history['num_sampled'].append(total_sampled)
                    history['num_unique_solutions'].append(unique_solutions)
                    history['mean_objectives'].append(mean_obj.tolist())

                    logger.info(
                        f"Iteration {iteration+1}/{num_iterations}: "
                        f"Sampled {total_sampled}, Terminal {num_terminal}, "
                        f"Unique {unique_solutions}, Mean Obj {mean_obj}"
                    )

        logger.info(
            f"Random sampling complete: {total_sampled} sampled, "
            f"{len(self.objectives_history)} terminal solutions"
        )

        return history

    def sample(
        self,
        num_samples: int,
        preference: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample solutions (ignores preference for random baseline).

        Args:
            num_samples: Number of trajectories to sample
            preference: Ignored (random sampler doesn't use preferences)

        Returns:
            Tuple of (objectives, states) arrays
            - objectives: (num_samples, num_objectives)
            - states: List of terminal states
        """
        if preference is not None:
            logger.warning("RandomSampler ignores preference parameter")

        sampled_objectives = []
        sampled_states = []

        for _ in range(num_samples):
            traj = self.sample_trajectory()
            if traj.is_terminal:
                sampled_objectives.append(traj.objectives)
                sampled_states.append(traj.states[-1])

        if len(sampled_objectives) == 0:
            logger.warning("No terminal trajectories sampled")
            return np.array([]), []

        return np.array(sampled_objectives), sampled_states

    def get_pareto_front(self, epsilon: float = 1e-9) -> np.ndarray:
        """
        Extract Pareto front from sampled trajectories.

        Args:
            epsilon: Small value for numerical stability

        Returns:
            Pareto front objectives array of shape (num_pareto, num_objectives)
        """
        if len(self.objectives_history) == 0:
            logger.warning("No trajectories sampled yet")
            return np.array([])

        objectives = np.array(self.objectives_history)
        return self._compute_pareto_front(objectives, epsilon)

    def _compute_pareto_front(
        self,
        objectives: np.ndarray,
        epsilon: float = 1e-9
    ) -> np.ndarray:
        """
        Compute Pareto front from objectives (maximization assumed).

        Args:
            objectives: Array of shape (num_solutions, num_objectives)
            epsilon: Tolerance for dominance comparison

        Returns:
            Pareto front array
        """
        is_pareto = np.ones(len(objectives), dtype=bool)

        for i, obj_i in enumerate(objectives):
            if not is_pareto[i]:
                continue

            # Check if any other solution dominates obj_i
            # Dominates if: all objectives >= obj_i and at least one > obj_i
            dominates = np.all(objectives >= obj_i - epsilon, axis=1) & \
                       np.any(objectives > obj_i + epsilon, axis=1)

            # Solution i is dominated if any solution dominates it
            if np.any(dominates):
                is_pareto[i] = False

        pareto_front = objectives[is_pareto]
        logger.info(f"Pareto front: {len(pareto_front)}/{len(objectives)} solutions")

        return pareto_front

    def get_all_objectives(self) -> np.ndarray:
        """
        Get all sampled objectives.

        Returns:
            Array of shape (num_trajectories, num_objectives)
        """
        if len(self.objectives_history) == 0:
            return np.array([])
        return np.array(self.objectives_history)

    def reset(self):
        """Clear all sampled trajectories."""
        self.trajectories.clear()
        self.objectives_history.clear()
        logger.info("RandomSampler reset")

    def __repr__(self) -> str:
        return (
            f"RandomSampler(env={self.env.__class__.__name__}, "
            f"max_steps={self.max_steps}, "
            f"num_sampled={len(self.trajectories)})"
        )