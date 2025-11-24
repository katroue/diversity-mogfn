"""
Hypernetwork-GFlowNet (HN-GFN) Baseline.

Implements the approach from:
    Zhu et al. "Sample-efficient Multi-objective Molecular Optimization with GFlowNets" (NeurIPS 2023)
    Paper: https://arxiv.org/abs/2302.04040
    Code: https://github.com/violet-sto/HN-GFN

Key innovation: Uses a preference-conditioned hypernetwork for the log partition function Z
instead of a fixed learned parameter. This allows Z to vary based on the preference vector,
improving multi-objective optimization.

Key differences from MOGFN-PC:
- MOGFN-PC: Fixed log_Z parameter (scalar)
- HN-GFN: Hypernetwork Z(preference) that outputs preference-dependent log partition function
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

# Import from parent package
import sys
import os
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from models.mogfn_pc import MOGFN_PC, PreferenceEncoder, ConditionalPolicyNetwork
from models.gflownet import Trajectory

logger = logging.getLogger(__name__)


class LogZHypernetwork(nn.Module):
    """
    Hypernetwork for computing preference-dependent log partition function.

    Instead of learning a fixed log_Z parameter, this network takes a preference
    vector as input and outputs a preference-dependent log partition function.

    Architecture:
        preference → [Linear] → ReLU → [Linear] → ReLU → [Linear] → log_Z(preference)
    """

    def __init__(self, num_objectives: int, hidden_dim: int = 64, num_layers: int = 3):
        """
        Args:
            num_objectives: Dimension of preference vector
            hidden_dim: Hidden layer dimension (default: 64)
            num_layers: Number of layers (default: 3)
        """
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(num_objectives, hidden_dim))
        layers.append(nn.LeakyReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())

        # Output layer (scalar log_Z)
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, preference: torch.Tensor) -> torch.Tensor:
        """
        Compute preference-dependent log partition function.

        Args:
            preference: Preference vector, shape (num_objectives,) or (batch, num_objectives)

        Returns:
            log_Z: Preference-dependent log partition function, shape () or (batch,)
        """
        output = self.network(preference)

        # Squeeze to scalar if single sample
        if output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output


@dataclass
class HNGFNTrajectory:
    """Trajectory for HN-GFN training."""
    states: List[torch.Tensor]
    actions: List[int]
    objectives: np.ndarray  # Shape: (num_objectives,)
    preference: np.ndarray  # Shape: (num_objectives,)
    log_probs: List[torch.Tensor]
    is_terminal: bool


class HN_GFN:
    """
    Hypernetwork-GFlowNet baseline for multi-objective optimization.

    Uses a preference-conditioned hypernetwork for the log partition function Z,
    allowing Z to adapt based on the preference vector. This improves sample
    efficiency in multi-objective settings compared to fixed log_Z.

    Training procedure:
    1. Sample preference from Dirichlet distribution
    2. Sample trajectory using preference-conditional policy
    3. Compute reward as weighted sum: r = preference · objectives
    4. Compute preference-dependent log_Z using hypernetwork: log_Z = Z(preference)
    5. Train with trajectory balance loss using Z(preference)

    Args:
        env: MultiObjectiveEnvironment instance
        state_dim: Dimension of state space
        num_objectives: Number of objectives
        hidden_dim: Hidden dimension for networks
        num_actions: Number of possible actions
        num_layers: Number of layers in policy networks (default: 3)
        z_hidden_dim: Hidden dimension for Z hypernetwork (default: 64)
        z_num_layers: Number of layers in Z hypernetwork (default: 3)
        preference_encoding: 'vanilla' or 'thermometer' (default: 'vanilla')
        conditioning_type: 'concat' or 'film' (default: 'concat')
        learning_rate: Learning rate for policy (default: 1e-3)
        z_learning_rate: Learning rate for Z hypernetwork (default: 1e-3)
        alpha: Dirichlet concentration parameter (default: 1.5)
        max_steps: Maximum steps per trajectory (default: 100)
        temperature: Temperature for action sampling (default: 1.0)
        epsilon_clip: Minimum value for numerical stability (default: 1e-8)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        env,
        state_dim: int,
        num_objectives: int,
        hidden_dim: int,
        num_actions: int,
        num_layers: int = 3,
        z_hidden_dim: int = 64,
        z_num_layers: int = 3,
        preference_encoding: str = 'vanilla',
        conditioning_type: str = 'concat',
        learning_rate: float = 1e-3,
        z_learning_rate: float = 1e-3,
        alpha: float = 1.5,
        max_steps: int = 100,
        temperature: float = 1.0,
        epsilon_clip: float = 1e-8,
        seed: Optional[int] = None
    ):
        self.env = env
        self.state_dim = state_dim
        self.num_objectives = num_objectives
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.temperature = temperature
        self.epsilon_clip = epsilon_clip
        self.alpha = alpha

        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

        # Policy networks (same as MOGFN-PC but without log_Z parameter)
        self.model = MOGFN_PC(
            state_dim=state_dim,
            num_objectives=num_objectives,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_layers=num_layers,
            preference_encoding=preference_encoding,
            conditioning_type=conditioning_type,
            temperature=temperature,
            sampling_strategy='categorical'
        )

        # Remove the fixed log_Z parameter from MOGFN-PC
        del self.model.log_Z

        # Add preference-dependent Z hypernetwork
        self.Z_network = LogZHypernetwork(
            num_objectives=num_objectives,
            hidden_dim=z_hidden_dim,
            num_layers=z_num_layers
        )

        # Separate optimizers for policy and Z hypernetwork
        self.policy_optimizer = optim.Adam(
            [p for name, p in self.model.named_parameters()],
            lr=learning_rate
        )
        self.z_optimizer = optim.Adam(
            self.Z_network.parameters(),
            lr=z_learning_rate
        )

        # Storage
        self.trajectories: List[HNGFNTrajectory] = []
        self.objectives_history: List[np.ndarray] = []
        self.training_losses: List[float] = []

        logger.info(
            f"Initialized HN_GFN with hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"z_hidden_dim={z_hidden_dim}, z_num_layers={z_num_layers}, "
            f"lr={learning_rate}, z_lr={z_learning_rate}, alpha={alpha}, seed={seed}"
        )

    def sample_preference(self) -> np.ndarray:
        """
        Sample preference from Dirichlet distribution.

        Returns:
            Preference vector of shape (num_objectives,), sums to 1
        """
        preference = self.rng.dirichlet(np.ones(self.num_objectives) * self.alpha)
        return preference

    def sample_trajectory(
        self,
        preference: Optional[np.ndarray] = None,
        explore: bool = True
    ) -> HNGFNTrajectory:
        """
        Sample a trajectory using preference-conditional policy.

        Args:
            preference: Preference vector (if None, samples from Dirichlet)
            explore: Whether to use exploration

        Returns:
            HNGFNTrajectory with states, actions, objectives, and preference
        """
        if preference is None:
            preference = self.sample_preference()

        preference_tensor = torch.FloatTensor(preference)

        state = self.env.get_initial_state()
        states = [state]
        actions = []
        log_probs = []
        is_done = False

        for step in range(self.max_steps):
            # Get valid actions
            valid_actions = self.env.get_valid_actions(state)
            if len(valid_actions) == 0:
                logger.warning(f"No valid actions at step {step}")
                break

            # Sample action from preference-conditional policy
            with torch.no_grad():
                action, log_prob = self.model.sample_action(
                    state=state,
                    preference=preference_tensor,
                    valid_actions=valid_actions,
                    explore=explore
                )

            actions.append(action)
            log_probs.append(log_prob)

            # Take action
            next_state, done = self.env.step(state, action)
            state = next_state
            states.append(state)
            is_done = done

            if done:
                break

        # Get final objectives
        objectives_tensor = self.env.compute_objectives(state)
        objectives = (
            objectives_tensor.detach().cpu().numpy()
            if torch.is_tensor(objectives_tensor)
            else np.array(objectives_tensor)
        )

        trajectory = HNGFNTrajectory(
            states=states,
            actions=actions,
            objectives=objectives,
            preference=preference,
            log_probs=log_probs,
            is_terminal=is_done
        )

        return trajectory

    def compute_trajectory_balance_loss(
        self,
        trajectory: HNGFNTrajectory
    ) -> torch.Tensor:
        """
        Compute trajectory balance loss with preference-dependent Z.

        Key difference from MOGFN-PC: Uses Z(preference) instead of fixed log_Z.

        Trajectory balance: log Z(w) + sum log P_F = log R(w) + sum log P_B
        where:
            - Z(w) = hypernetwork output for preference w
            - R(w) = weighted reward: w · objectives
            - P_F = forward policy
            - P_B = backward policy

        Args:
            trajectory: HNGFNTrajectory with preference

        Returns:
            Loss tensor (MSE of trajectory balance equation)
        """
        if not trajectory.is_terminal:
            return torch.tensor(0.0)

        # Convert preference to tensor
        preference = torch.FloatTensor(trajectory.preference)

        # Compute preference-dependent log_Z using hypernetwork
        log_Z = self.Z_network(preference)

        # Compute forward log probabilities
        forward_log_probs = []
        for i, (state, action) in enumerate(zip(trajectory.states[:-1], trajectory.actions)):
            logits = self.model.forward_logits(state, preference)
            log_probs = torch.log_softmax(logits / self.temperature, dim=-1)
            forward_log_probs.append(log_probs[action])

        # Compute backward log probabilities
        backward_log_probs = []
        for i, (state, action) in enumerate(zip(trajectory.states[1:], trajectory.actions)):
            logits = self.model.backward_logits(state, preference)
            log_probs = torch.log_softmax(logits / self.temperature, dim=-1)
            backward_log_probs.append(log_probs[action])

        # Sum log probabilities
        sum_forward = sum(forward_log_probs)
        sum_backward = sum(backward_log_probs) if len(backward_log_probs) > 0 else torch.tensor(0.0)

        # Compute weighted reward: R(w) = w · objectives
        reward = np.dot(trajectory.objectives, trajectory.preference)
        log_R = torch.log(torch.tensor(reward + self.epsilon_clip))

        # Trajectory balance loss: (log Z(w) + sum log P_F) - (log R(w) + sum log P_B)
        loss = (log_Z + sum_forward - log_R - sum_backward) ** 2

        return loss

    def train(
        self,
        num_iterations: int,
        batch_size: int = 32,
        log_interval: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train HN-GFN with preference-dependent Z hypernetwork.

        Args:
            num_iterations: Number of training iterations
            batch_size: Number of trajectories per batch
            log_interval: Log progress every N iterations
            **kwargs: Additional arguments (for API compatibility)

        Returns:
            Training history dictionary
        """
        logger.info(
            f"Starting HN-GFN training: {num_iterations} iterations × {batch_size} batch size"
        )

        self.model.train()
        self.Z_network.train()
        total_sampled = 0

        history = {
            'iteration': [],
            'loss': [],
            'num_terminal': [],
            'num_unique_solutions': [],
            'mean_objectives': []
        }

        for iteration in range(num_iterations):
            batch_losses = []

            for _ in range(batch_size):
                # Sample preference from Dirichlet
                preference = self.sample_preference()

                # Sample trajectory conditioned on preference
                traj = self.sample_trajectory(preference=preference, explore=True)

                if traj.is_terminal:
                    # Compute loss with preference-dependent Z
                    loss = self.compute_trajectory_balance_loss(traj)
                    batch_losses.append(loss)

                    # Store trajectory
                    self.trajectories.append(traj)
                    self.objectives_history.append(traj.objectives)

            total_sampled += batch_size

            # Update model
            if len(batch_losses) > 0:
                total_loss = sum(batch_losses) / len(batch_losses)

                # Update policy networks
                self.policy_optimizer.zero_grad()
                # Update Z hypernetwork
                self.z_optimizer.zero_grad()

                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.Z_network.parameters(), max_norm=1.0)

                self.policy_optimizer.step()
                self.z_optimizer.step()

                self.training_losses.append(total_loss.item())

            # Memory management: Keep only recent trajectories (prevents OOM on long runs)
            MAX_TRAJECTORIES_IN_MEMORY = 10000
            if len(self.trajectories) > MAX_TRAJECTORIES_IN_MEMORY:
                # Keep only the most recent trajectories
                self.trajectories = self.trajectories[-MAX_TRAJECTORIES_IN_MEMORY:]
                self.objectives_history = self.objectives_history[-MAX_TRAJECTORIES_IN_MEMORY:]
                # Force garbage collection
                import gc
                gc.collect()

            # Log progress
            if (iteration + 1) % log_interval == 0 or iteration == 0:
                num_terminal = len(self.objectives_history)

                if num_terminal > 0:
                    unique_solutions = len(set(tuple(obj) for obj in self.objectives_history))
                    mean_obj = np.mean(self.objectives_history, axis=0)
                    avg_loss = np.mean(self.training_losses[-log_interval:]) if len(self.training_losses) > 0 else 0.0

                    history['iteration'].append(iteration + 1)
                    history['loss'].append(avg_loss)
                    history['num_terminal'].append(num_terminal)
                    history['num_unique_solutions'].append(unique_solutions)
                    history['mean_objectives'].append(mean_obj.tolist())

                    logger.info(
                        f"Iteration {iteration+1}/{num_iterations}: "
                        f"Loss {avg_loss:.4f}, Terminal {num_terminal}, "
                        f"Unique {unique_solutions}, Mean Obj {mean_obj}"
                    )

        logger.info(
            f"HN-GFN training complete: {total_sampled} sampled, "
            f"{len(self.objectives_history)} terminal solutions"
        )

        return history

    def sample(
        self,
        num_samples: int,
        preference: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List]:
        """
        Sample solutions using the trained model.

        If preference is provided, sample conditioned on that preference.
        Otherwise, sample with random preferences for diversity.

        Args:
            num_samples: Number of trajectories to sample
            preference: Optional preference vector (shape: num_objectives,)

        Returns:
            Tuple of (objectives, states)
            - objectives: (num_samples, num_objectives)
            - states: List of terminal states
        """
        self.model.eval()
        self.Z_network.eval()

        sampled_objectives = []
        sampled_states = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Use provided preference or sample random one
                if preference is not None:
                    pref = preference
                else:
                    pref = self.sample_preference()

                pref_tensor = torch.FloatTensor(pref)

                # Sample trajectory
                state = self.env.get_initial_state()

                for step in range(self.max_steps):
                    valid_actions = self.env.get_valid_actions(state)
                    if len(valid_actions) == 0:
                        break

                    action, _ = self.model.sample_action(
                        state=state,
                        preference=pref_tensor,
                        valid_actions=valid_actions,
                        explore=False
                    )

                    next_state, done = self.env.step(state, action)
                    state = next_state

                    if done:
                        break

                # Get objectives
                objectives_tensor = self.env.compute_objectives(state)
                objectives = (
                    objectives_tensor.detach().cpu().numpy()
                    if torch.is_tensor(objectives_tensor)
                    else np.array(objectives_tensor)
                )

                sampled_objectives.append(objectives)
                sampled_states.append(state)

        self.model.train()
        self.Z_network.train()

        return np.array(sampled_objectives), sampled_states

    def get_pareto_front(self, epsilon: float = 1e-9) -> np.ndarray:
        """
        Extract Pareto front from sampled trajectories.

        Args:
            epsilon: Tolerance for dominance comparison

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
            dominates = (
                np.all(objectives >= obj_i - epsilon, axis=1) &
                np.any(objectives > obj_i + epsilon, axis=1)
            )

            if np.any(dominates):
                is_pareto[i] = False

        pareto_front = objectives[is_pareto]
        logger.info(f"Pareto front: {len(pareto_front)}/{len(objectives)} solutions")

        return pareto_front

    def get_all_objectives(self) -> np.ndarray:
        """
        Get all sampled objectives from training.

        Returns:
            Array of shape (num_trajectories, num_objectives)
        """
        if len(self.objectives_history) == 0:
            return np.array([])
        return np.array(self.objectives_history)

    def reset(self):
        """Clear trajectories and reset model."""
        self.trajectories.clear()
        self.objectives_history.clear()
        self.training_losses.clear()

        # Reinitialize model weights
        def reset_weights(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.model.apply(reset_weights)
        self.Z_network.apply(reset_weights)
        logger.info("HN_GFN reset")

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'z_network_state_dict': self.Z_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'z_optimizer_state_dict': self.z_optimizer.state_dict(),
            'objectives_history': self.objectives_history,
            'training_losses': self.training_losses
        }, path)
        logger.info(f"Saved HN_GFN checkpoint to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        # Use weights_only=False for compatibility with numpy arrays in checkpoint
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.Z_network.load_state_dict(checkpoint['z_network_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.z_optimizer.load_state_dict(checkpoint['z_optimizer_state_dict'])
        self.objectives_history = checkpoint['objectives_history']
        self.training_losses = checkpoint['training_losses']
        logger.info(f"Loaded HN_GFN checkpoint from {path}")

    def __repr__(self) -> str:
        return (
            f"HN_GFN(env={self.env.__class__.__name__}, "
            f"state_dim={self.state_dim}, "
            f"num_objectives={self.num_objectives}, "
            f"num_sampled={len(self.trajectories)})"
        )
