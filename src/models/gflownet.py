"""
Base GFlowNet implementation.

This module provides the foundational GFlowNet class with trajectory balance
loss and basic sampling functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
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


class PolicyNetwork(nn.Module):
    """Forward policy network for GFlowNet."""
    
    def __init__(self, 
                state_dim: int,
                hidden_dim: int,
                num_actions: int,
                num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits for given state.
        
        Args:
            state: State tensor, shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            logits: Action logits, shape (batch_size, num_actions) or (num_actions,)
        """
        return self.network(state)


class BackwardPolicyNetwork(nn.Module):
    """Backward policy network for GFlowNet."""
    
    def __init__(self,
                state_dim: int,
                hidden_dim: int,
                num_actions: int,
                num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class BaseGFlowNet(nn.Module):
    """
    Base GFlowNet with Trajectory Balance loss.
    
    Based on:
        Malkin et al. "Trajectory Balance: Improved Credit Assignment in GFlowNets" (2022)
    """
    
    def __init__(self,
                state_dim: int,
                hidden_dim: int,
                num_actions: int,
                num_layers: int = 3,
                exploration_rate: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.exploration_rate = exploration_rate
        
        # Forward policy
        self.forward_policy = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_layers=num_layers
        )
        
        # Backward policy
        self.backward_policy = BackwardPolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_layers=num_layers
        )
        
        # Log Z (partition function estimate) - learnable parameter
        self.log_Z = nn.Parameter(torch.zeros(1))
    
    def forward_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get forward policy logits."""
        return self.forward_policy(state)
    
    def backward_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get backward policy logits."""
        return self.backward_policy(state)
    
    def sample_action(self, 
                    state: torch.Tensor,
                    valid_actions: Optional[List[int]] = None,
                    explore: bool = True) -> Tuple[int, torch.Tensor]:
        """
        Sample action from forward policy.
        
        Args:
            state: Current state
            valid_actions: List of valid action indices (None = all valid)
            explore: Whether to use epsilon-greedy exploration
        
        Returns:
            action: Sampled action index
            log_prob: Log probability of sampled action
        """
        logits = self.forward_logits(state)
        
        # Mask invalid actions
        if valid_actions is not None:
            mask = torch.full_like(logits, float('-inf'))
            mask[valid_actions] = 0
            logits = logits + mask
        
        # Epsilon-greedy exploration
        if explore and torch.rand(1).item() < self.exploration_rate:
            if valid_actions is None:
                valid_actions = list(range(self.num_actions))
            action = torch.tensor(valid_actions[torch.randint(len(valid_actions), (1,)).item()])
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze()
        
        log_prob = F.log_softmax(logits, dim=-1)[action]
        
        return action.item(), log_prob
    
    def trajectory_balance_loss(self, 
                                trajectories: List[Trajectory]) -> torch.Tensor:
        """
        Compute trajectory balance loss.
        
        TB loss: (log Z + sum log P_F - log R - sum log P_B)^2
        
        Args:
            trajectories: List of sampled trajectories
        
        Returns:
            loss: Trajectory balance loss
        """
        losses = []
        
        for traj in trajectories:
            # Forward flow: log Z + sum_t log P_F(a_t | s_t)
            log_forward_flow = self.log_Z
            for state, action in zip(traj.states[:-1], traj.actions):
                logits = self.forward_logits(state)
                log_prob = F.log_softmax(logits, dim=-1)[action]
                log_forward_flow = log_forward_flow + log_prob
            
            # Backward flow: log R(x) + sum_t log P_B(a_t | s_t)
            log_reward = torch.log(torch.tensor(traj.reward) + 1e-10)
            log_backward_flow = log_reward
            
            for state, action in zip(traj.states[1:], traj.actions):
                logits = self.backward_logits(state)
                log_prob = F.log_softmax(logits, dim=-1)[action]
                log_backward_flow = log_backward_flow + log_prob
            
            # TB loss: (log_forward - log_backward)^2
            loss = (log_forward_flow - log_backward_flow) ** 2
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def detailed_balance_loss(self, 
                            trajectories: List[Trajectory]) -> torch.Tensor:
        """
        Compute detailed balance loss (alternative to TB).
        
        DB loss: sum_t (log F(s_t) + log P_F(a_t|s_t) - log F(s_{t+1}) - log P_B(a_t|s_{t+1}))^2
        """
        # Simplified implementation - would need state flow estimation
        raise NotImplementedError("Use trajectory_balance_loss instead")
    
    def compute_loss(self,
                    trajectories: List[Trajectory],
                    loss_type: str = 'tb') -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            trajectories: List of sampled trajectories
            loss_type: 'tb' for trajectory balance
        
        Returns:
            loss: Training loss
        """
        if loss_type == 'tb':
            return self.trajectory_balance_loss(trajectories)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class GFlowNetEnvironment(ABC):
    """Abstract base class for GFlowNet environments."""
    
    @abstractmethod
    def get_initial_state(self) -> torch.Tensor:
        """Get initial state."""
        pass
    
    @abstractmethod
    def step(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, bool]:
        """
        Take action in environment.
        
        Returns:
            next_state: Next state after action
            is_terminal: Whether next_state is terminal
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, state: torch.Tensor) -> List[int]:
        """Get list of valid actions for state."""
        pass
    
    @abstractmethod
    def compute_reward(self, state: torch.Tensor) -> float:
        """Compute reward for terminal state."""
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


class GFlowNetSampler:
    """Sampler for generating trajectories from GFlowNet."""
    
    def __init__(self, 
                gflownet: BaseGFlowNet,
                env: GFlowNetEnvironment):
        self.gflownet = gflownet
        self.env = env
    
    def sample_trajectory(self, explore: bool = True) -> Trajectory:
        """
        Sample a complete trajectory from initial state to terminal state.
        
        Args:
            explore: Whether to use exploration
        
        Returns:
            trajectory: Complete trajectory
        """
        states = []
        actions = []
        log_probs = []
        
        # Start from initial state
        state = self.env.get_initial_state()
        states.append(state)
        is_terminal = False
        
        # Sample until terminal
        while not is_terminal:
            valid_actions = self.env.get_valid_actions(state)
            
            # Sample action
            action, log_prob = self.gflownet.sample_action(
                state, valid_actions, explore=explore
            )
            
            # Take step
            next_state, is_terminal = self.env.step(state, action)
            
            # Record
            actions.append(action)
            log_probs.append(log_prob)
            states.append(next_state)
            
            state = next_state
        
        # Compute reward for terminal state
        reward = self.env.compute_reward(state)
        
        return Trajectory(
            states=states,
            actions=actions,
            log_probs=log_probs,
            is_terminal=True,
            reward=reward
        )
    
    def sample_batch(self, batch_size: int, explore: bool = True) -> List[Trajectory]:
        """Sample a batch of trajectories."""
        return [self.sample_trajectory(explore) for _ in range(batch_size)]


def test_gflownet():
    """Simple test for GFlowNet implementation."""
    
    # Create simple test environment (mock)
    class DummyEnv(GFlowNetEnvironment):
        def get_initial_state(self):
            return torch.zeros(4)
        
        def step(self, state, action):
            next_state = state.clone()
            next_state[action] = 1.0
            is_terminal = torch.sum(next_state) >= 3
            return next_state, is_terminal
        
        def get_valid_actions(self, state):
            return [i for i in range(4) if state[i] == 0]
        
        def compute_reward(self, state):
            return torch.sum(state).item()
        
        @property
        def state_dim(self):
            return 4
        
        @property
        def num_actions(self):
            return 4
    
    # Create GFlowNet
    env = DummyEnv()
    gfn = BaseGFlowNet(
        state_dim=env.state_dim,
        hidden_dim=64,
        num_actions=env.num_actions
    )
    
    # Sample trajectories
    sampler = GFlowNetSampler(gfn, env)
    trajectories = sampler.sample_batch(4)
    
    # Compute loss
    loss = gfn.compute_loss(trajectories)
    
    print(f"GFlowNet test passed! Loss: {loss.item():.4f}")
    print(f"Sampled {len(trajectories)} trajectories")
    print(f"Trajectory lengths: {[len(t) for t in trajectories]}")


if __name__ == '__main__':
    test_gflownet()
