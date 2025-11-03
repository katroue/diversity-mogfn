"""
Multi-Objective GFlowNet (MOGFN) implementation.

Implements MOGFN-PC (Preference-Conditional) from:
    Jain et al. "Multi-Objective GFlowNets" (ICML 2023)

This module extends BaseGFlowNet to handle multiple objectives through
preference-conditional policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from abc import ABC, abstractmethod

try:
    # Normal package import when used as part of the `models` package
    from .gflownet import BaseGFlowNet, Trajectory, GFlowNetEnvironment, PolicyNetwork
except Exception:
    # Fallback for running this file directly (python src/models/mogfn_pc.py)
    # Add the `src` directory to sys.path so `models` becomes importable.
    import os, sys

    src_models_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/models
    src_dir = os.path.dirname(src_models_dir)  # .../src
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from models.gflownet import BaseGFlowNet, Trajectory, GFlowNetEnvironment, PolicyNetwork


class PreferenceEncoder(nn.Module):
    """
    Encodes preference vectors for conditioning.
    
    Supports both vanilla encoding and thermometer encoding.
    """
    
    def __init__(self, 
                num_objectives: int,
                encoding_type: str = 'vanilla',
                output_dim: Optional[int] = None):
        """
        Args:
            num_objectives: Number of objectives
            encoding_type: 'vanilla' or 'thermometer'
            output_dim: Output dimension (None = same as input)
        """
        super().__init__()
        self.num_objectives = num_objectives
        self.encoding_type = encoding_type
        
        if encoding_type == 'thermometer':
            # Thermometer encoding: map [0,1] to discretized bins
            self.num_bins = 10
            input_dim = num_objectives * self.num_bins
        else:
            input_dim = num_objectives
        
        if output_dim is not None:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = None
            output_dim = input_dim
        
        self.output_dim = output_dim
    
    def forward(self, preference: torch.Tensor) -> torch.Tensor:
        """
        Encode preference vector.
        
        Args:
            preference: Preference vector, shape (num_objectives,) or (batch, num_objectives)
        
        Returns:
            encoded: Encoded preference, shape (output_dim,) or (batch, output_dim)
        """
        if self.encoding_type == 'thermometer':
            encoded = self._thermometer_encode(preference)
        else:
            encoded = preference
        
        if self.projection is not None:
            encoded = self.projection(encoded)
        
        return encoded
    
    def _thermometer_encode(self, preference: torch.Tensor) -> torch.Tensor:
        """Convert preference to thermometer encoding."""
        # For each objective value in [0,1], create binary vector
        # e.g., value=0.35 with 10 bins -> [1,1,1,0,0,0,0,0,0,0]
        batch_shape = preference.shape[:-1]
        
        bins = torch.linspace(0, 1, self.num_bins + 1, device=preference.device)[1:]
        bins = bins.view(*([1] * len(batch_shape)), 1, -1)
        
        pref_expanded = preference.unsqueeze(-1)
        thermometer = (pref_expanded >= bins).float()
        
        # Flatten last two dimensions
        return thermometer.reshape(*batch_shape, -1)


class ConditionalPolicyNetwork(nn.Module):
    """Policy network conditioned on preference vector."""
    
    def __init__(self,
                state_dim: int,
                preference_dim: int,
                hidden_dim: int,
                num_actions: int,
                num_layers: int = 3,
                conditioning_type: str = 'concat'):
        """
        Args:
            conditioning_type: 'concat', 'film'
        """
        super().__init__()
        
        self.conditioning_type = conditioning_type
        
        if conditioning_type == 'concat':
            input_dim = state_dim + preference_dim
            self.network = self._build_mlp(input_dim, hidden_dim, num_actions, num_layers)
        
        elif conditioning_type == 'film':
            # FiLM: Feature-wise Linear Modulation
            self.state_encoder = self._build_mlp(state_dim, hidden_dim, hidden_dim, num_layers - 1)
            
            # FiLM conditioning: generates scale and shift parameters
            self.film_layer = nn.Linear(preference_dim, 2 * hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, num_actions)
        
        else:
            raise NotImplementedError(f"Conditioning type {conditioning_type} not implemented")
    
    def _build_mlp(self, input_dim, hidden_dim, output_dim, num_layers):
        """Helper to build MLP."""
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, 
                state: torch.Tensor, 
                preference: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits conditioned on preference.
        
        Args:
            state: State tensor
            preference: Encoded preference vector
        
        Returns:
            logits: Action logits
        """
        if self.conditioning_type == 'concat':
            combined = torch.cat([state, preference], dim=-1)
            return self.network(combined)
        
        elif self.conditioning_type == 'film':
            # Encode state
            features = self.state_encoder(state)
            
            # Generate FiLM parameters
            film_params = self.film_layer(preference)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            
            # Apply FiLM: scale and shift
            features = gamma * features + beta
            
            # Output layer
            return self.output_layer(features)


class MOGFN_PC(BaseGFlowNet):
    """
    Multi-Objective GFlowNet with Preference Conditioning (MOGFN-PC).
    
    Extends BaseGFlowNet to handle multiple objectives by conditioning
    the policy on preference vectors.
    """
    
    def __init__(self,
                state_dim: int,
                num_objectives: int,
                hidden_dim: int,
                num_actions: int,
                num_layers: int = 3,
                preference_encoding: str = 'vanilla',
                conditioning_type: str = 'concat',
                exploration_rate: float = 0.1,
                temperature: float = 1.0,
                sampling_strategy: str = 'categorical',
                top_k: Optional[int] = None,
                top_p: Optional[float] = None):
        """
        Args:
            state_dim: Dimension of state space
            num_objectives: Number of objectives
            hidden_dim: Hidden dimension for networks
            num_actions: Number of possible actions
            num_layers: Number of layers in networks
            preference_encoding: 'vanilla' or 'thermometer'
            conditioning_type: 'concat' or 'film'
            exploration_rate: Epsilon for exploration
            temperature: Temperature for softmax (higher = more random)
            sampling_strategy: 'greedy', 'categorical', 'top_k', or 'nucleus'
            top_k: Number of top actions to sample from (for top_k strategy)
            top_p: Probability mass to sample from (for nucleus strategy)
        """
        # Don't call super().__init__ since we're replacing the networks
        nn.Module.__init__(self)

        self.state_dim = state_dim
        self.num_objectives = num_objectives
        self.num_actions = num_actions
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.sampling_strategy = sampling_strategy
        self.top_k = top_k
        self.top_p = top_p
        
        # Preference encoder
        self.preference_encoder = PreferenceEncoder(
            num_objectives=num_objectives,
            encoding_type=preference_encoding,
            output_dim=hidden_dim if preference_encoding == 'thermometer' else None
        )
        
        preference_dim = self.preference_encoder.output_dim
        
        # Conditional forward policy
        self.forward_policy = ConditionalPolicyNetwork(
            state_dim=state_dim,
            preference_dim=preference_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_layers=num_layers,
            conditioning_type=conditioning_type
        )
        
        # Conditional backward policy
        self.backward_policy = ConditionalPolicyNetwork(
            state_dim=state_dim,
            preference_dim=preference_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_layers=num_layers,
            conditioning_type=conditioning_type
        )
        
        # Log Z - now may depend on preference
        self.log_Z = nn.Parameter(torch.zeros(1))
    
    def forward_logits(self, 
                    state: torch.Tensor,
                    preference: torch.Tensor) -> torch.Tensor:
        """Get forward policy logits conditioned on preference."""
        encoded_pref = self.preference_encoder(preference)
        return self.forward_policy(state, encoded_pref)
    
    def backward_logits(self,
                    state: torch.Tensor,
                    preference: torch.Tensor) -> torch.Tensor:
        """Get backward policy logits conditioned on preference."""
        encoded_pref = self.preference_encoder(preference)
        return self.backward_policy(state, encoded_pref)
    
    def sample_action(self,
                    state: torch.Tensor,
                    preference: torch.Tensor,
                    valid_actions: Optional[List[int]] = None,
                    explore: bool = True) -> Tuple[int, torch.Tensor]:
        """
        Sample action from preference-conditional policy.
        
        Args:
            state: Current state
            preference: Preference vector
            valid_actions: List of valid action indices
            explore: Whether to use exploration
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        logits = self.forward_logits(state, preference)
        
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
            # Apply temperature and use specified sampling strategy
            if self.sampling_strategy == 'greedy':
                # Greedy: select action with highest logit
                action = torch.argmax(logits)

            elif self.sampling_strategy == 'categorical':
                # Categorical: sample from softmax distribution with temperature
                probs = F.softmax(logits / self.temperature, dim=-1)
                action = torch.multinomial(probs, 1).squeeze()

            elif self.sampling_strategy == 'top_k':
                # Top-k: sample from top-k actions
                if self.top_k is not None and self.top_k > 0:
                    top_k = min(self.top_k, logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    probs = F.softmax(top_k_logits / self.temperature, dim=-1)
                    action_idx = torch.multinomial(probs, 1).squeeze()
                    action = top_k_indices[action_idx]
                else:
                    # Fallback to categorical if top_k not specified
                    probs = F.softmax(logits / self.temperature, dim=-1)
                    action = torch.multinomial(probs, 1).squeeze()

            elif self.sampling_strategy == 'nucleus':
                # Nucleus (top-p): sample from cumulative probability mass
                if self.top_p is not None and 0 < self.top_p < 1:
                    probs = F.softmax(logits / self.temperature, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Find cutoff index where cumulative prob exceeds top_p
                    cutoff_idx = torch.where(cumulative_probs > self.top_p)[0]
                    if len(cutoff_idx) > 0:
                        cutoff_idx = cutoff_idx[0].item() + 1
                    else:
                        cutoff_idx = len(sorted_probs)

                    # Keep only top-p probability mass
                    nucleus_probs = sorted_probs[:cutoff_idx]
                    nucleus_indices = sorted_indices[:cutoff_idx]
                    nucleus_probs = nucleus_probs / nucleus_probs.sum()  # Renormalize

                    action_idx = torch.multinomial(nucleus_probs, 1).squeeze()
                    action = nucleus_indices[action_idx]
                else:
                    # Fallback to categorical if top_p not specified
                    probs = F.softmax(logits / self.temperature, dim=-1)
                    action = torch.multinomial(probs, 1).squeeze()

            else:
                # Default to categorical
                probs = F.softmax(logits / self.temperature, dim=-1)
                action = torch.multinomial(probs, 1).squeeze()

        # Compute log probability with temperature applied
        log_prob = F.log_softmax(logits / self.temperature, dim=-1)[action]
        
        return action.item(), log_prob
    
    def compute_scalarized_reward(self,
                                objectives: torch.Tensor,
                                preference: torch.Tensor,
                                beta: float = 1.0) -> torch.Tensor:
        """
        Compute scalarized reward using preference vector.
        
        R(x|ω) = (Σ ωᵢ * Rᵢ(x))^β, best performance according to the original paper
        
        Args:
            objectives: Objective values, shape (num_objectives,)
            preference: Preference vector, shape (num_objectives,)
            beta: Temperature parameter (reward exponent)
        
        Returns:
            scalar_reward: Scalarized reward
        """
        weighted_sum = torch.sum(preference * objectives)
        return torch.pow(weighted_sum, beta)
    
    def trajectory_balance_loss(self,
                            trajectories: List[Trajectory],
                            preferences: List[torch.Tensor],
                            beta: float = 1.0,
                            log_reward_clip: float = 10.0) -> torch.Tensor:
        """
        Compute preference-conditional trajectory balance loss.

        Args:
            trajectories: List of trajectories
            preferences: List of preference vectors (one per trajectory)
            beta: Reward exponent
            log_reward_clip: Clipping value for log rewards

        Returns:
            loss: TB loss
        """
        losses = []

        for traj, pref in zip(trajectories, preferences):
            # Forward flow: log Z + sum_t log P_F(a_t | s_t, ω)
            log_forward_flow = self.log_Z

            for state, action in zip(traj.states[:-1], traj.actions):
                logits = self.forward_logits(state, pref)
                log_prob = F.log_softmax(logits, dim=-1)[action]
                log_forward_flow = log_forward_flow + log_prob

            # Backward flow: log R(x|ω) + sum_t log P_B(a_t | s_t, ω)
            # traj.reward should be the multi-objective rewards
            if isinstance(traj.reward, torch.Tensor):
                scalar_reward = self.compute_scalarized_reward(traj.reward, pref, beta)
            else:
                scalar_reward = torch.tensor(traj.reward, device=self.log_Z.device)

            log_reward = torch.clamp(torch.log(scalar_reward + 1e-10), max=log_reward_clip)
            log_backward_flow = log_reward

            for state, action in zip(traj.states[1:], traj.actions):
                logits = self.backward_logits(state, pref)
                log_prob = F.log_softmax(logits, dim=-1)[action]
                log_backward_flow = log_backward_flow + log_prob

            # TB loss
            loss = (log_forward_flow - log_backward_flow) ** 2
            losses.append(loss)

        return torch.stack(losses).mean()
    
    def detailed_balance_loss(self,
                             trajectories: List[Trajectory],
                             preferences: List[torch.Tensor],
                             beta: float = 1.0,
                             log_reward_clip: float = 10.0) -> torch.Tensor:
        """
        Compute detailed balance loss.

        Args:
            trajectories: List of trajectories
            preferences: List of preference vectors
            beta: Reward exponent
            log_reward_clip: Clipping value for log rewards

        Returns:
            loss: DB loss
        """
        losses = []

        for traj, pref in zip(trajectories, preferences):
            traj_loss = 0.0

            # Compute reward
            if isinstance(traj.reward, torch.Tensor):
                scalar_reward = self.compute_scalarized_reward(traj.reward, pref, beta)
            else:
                scalar_reward = torch.tensor(traj.reward)

            log_reward = torch.clamp(torch.log(scalar_reward + 1e-10), max=log_reward_clip)

            # Detailed balance: P_F(s'|s,ω) / P_B(s|s',ω) = R(s'|ω) / Z(ω)
            for t in range(len(traj.states) - 1):
                state = traj.states[t]
                next_state = traj.states[t + 1]
                action = traj.actions[t]

                # Forward transition
                forward_logits = self.forward_logits(state, pref)
                log_forward_prob = F.log_softmax(forward_logits, dim=-1)[action]

                # Backward transition
                backward_logits = self.backward_logits(next_state, pref)
                log_backward_prob = F.log_softmax(backward_logits, dim=-1)[action]

                # DB condition
                if t == len(traj.states) - 2:  # Terminal state
                    db_loss = (log_forward_prob - log_backward_prob - log_reward + self.log_Z) ** 2
                else:
                    db_loss = (log_forward_prob - log_backward_prob) ** 2

                traj_loss = traj_loss + db_loss

            losses.append(traj_loss)

        return torch.stack(losses).mean()

    def subtrajectory_balance_loss(self,
                                   trajectories: List[Trajectory],
                                   preferences: List[torch.Tensor],
                                   beta: float = 1.0,
                                   lambda_: float = 0.9,
                                   log_reward_clip: float = 10.0) -> torch.Tensor:
        """
        Compute sub-trajectory balance loss.

        SubTB samples sub-trajectories and applies TB loss to them.
        With lambda_=1.0, this is equivalent to full TB.
        With lambda_→0, this approaches DB (single-step).

        Args:
            trajectories: List of trajectories
            preferences: List of preference vectors
            beta: Reward exponent
            lambda_: Geometric distribution parameter (higher = longer sub-trajectories)
            log_reward_clip: Clipping value for log rewards

        Returns:
            loss: SubTB loss
        """
        losses = []

        for traj, pref in zip(trajectories, preferences):
            # Compute reward
            if isinstance(traj.reward, torch.Tensor):
                scalar_reward = self.compute_scalarized_reward(traj.reward, pref, beta)
            else:
                scalar_reward = torch.tensor(traj.reward, device=self.log_Z.device)

            log_reward = torch.clamp(torch.log(scalar_reward + 1e-10), max=log_reward_clip)

            traj_len = len(traj.states) - 1

            if traj_len == 0:
                continue  # Skip empty trajectories

            # Sample sub-trajectory using geometric distribution
            # Geometric(lambda_): P(length = k) ~ (1-lambda_)^k * lambda_
            # With lambda_ close to 1: prefer longer sub-trajectories
            # With lambda_ close to 0: prefer shorter sub-trajectories

            # For simplicity, use deterministic length based on lambda_
            # (In practice, could sample from geometric distribution)
            if lambda_ >= 1.0:
                # Full trajectory (standard TB)
                start_idx = 0
                end_idx = traj_len
            else:
                # Sample sub-trajectory length: geometric-like using lambda_
                # Mean length = lambda_ * traj_len
                import random

                # Sample uniformly from [1, traj_len]
                max_len = max(1, traj_len)

                # Use geometric sampling: favor longer for higher lambda_
                # Sample k ~ Geometric(1 - lambda_)
                # Then use min(k+1, traj_len) as length
                if lambda_ > 0:
                    # Expected length proportional to 1/(1-lambda_)
                    # Adjust to get mean ≈ lambda_ * traj_len
                    p_stop = 1.0 - lambda_
                    sampled_len = 1
                    while sampled_len < traj_len and random.random() > p_stop:
                        sampled_len += 1
                    subtraj_len = sampled_len
                else:
                    subtraj_len = 1  # Single step (like DB)

                # Sample random starting position
                max_start = max(0, traj_len - subtraj_len)
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                end_idx = start_idx + subtraj_len

            # Compute forward flow for sub-trajectory
            # If starting from s_0, include log Z
            # If starting from s_i (i>0), we'd need log F(s_i) but we approximate with 0
            if start_idx == 0:
                log_forward_flow = self.log_Z.clone()  # Clone to avoid in-place ops
            else:
                # For intermediate starting points, approximate initial flow as 0
                # (Ideally would use learned state values, but that requires V-function)
                log_forward_flow = torch.zeros_like(self.log_Z)

            for t in range(start_idx, end_idx):
                state = traj.states[t]
                action = traj.actions[t]
                forward_logits = self.forward_logits(state, pref)
                log_prob = F.log_softmax(forward_logits, dim=-1)[action]
                log_forward_flow = log_forward_flow + log_prob

            # Compute backward flow for sub-trajectory
            # If ending at terminal state s_T, include log R
            # If ending at s_j (j<T), we'd need log R(s_j) but approximate with 0
            if end_idx == traj_len:
                log_backward_flow = log_reward.clone()  # Clone to avoid in-place ops
            else:
                # For intermediate ending points, approximate terminal reward as 0
                log_backward_flow = torch.zeros_like(log_reward)

            for t in range(start_idx + 1, end_idx + 1):
                state = traj.states[t]
                action = traj.actions[t - 1]
                backward_logits = self.backward_logits(state, pref)
                log_prob = F.log_softmax(backward_logits, dim=-1)[action]
                log_backward_flow = log_backward_flow + log_prob

            # SubTB loss: balance forward and backward flow on sub-trajectory
            loss = (log_forward_flow - log_backward_flow) ** 2
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.log_Z.device, requires_grad=True)

        return torch.stack(losses).mean()

    def flow_matching_loss(self,
                          trajectories: List[Trajectory],
                          preferences: List[torch.Tensor],
                          beta: float = 1.0,
                          log_reward_clip: float = 10.0) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            trajectories: List of trajectories
            preferences: List of preference vectors
            beta: Reward exponent
            log_reward_clip: Clipping value for log rewards

        Returns:
            loss: FM loss
        """
        # Flow matching: match forward and backward flows at each state
        losses = []

        for traj, pref in zip(trajectories, preferences):
            # Compute reward
            if isinstance(traj.reward, torch.Tensor):
                scalar_reward = self.compute_scalarized_reward(traj.reward, pref, beta)
            else:
                scalar_reward = torch.tensor(traj.reward)

            log_reward = torch.clamp(torch.log(scalar_reward + 1e-10), max=log_reward_clip)

            traj_loss = 0.0

            # For each transition, match inflow and outflow
            for t in range(len(traj.states) - 1):
                state = traj.states[t]
                action = traj.actions[t]

                # Forward flow
                forward_logits = self.forward_logits(state, pref)
                log_forward_prob = F.log_softmax(forward_logits, dim=-1)[action]

                # Inflow to state
                if t == 0:
                    log_inflow = self.log_Z
                else:
                    prev_state = traj.states[t - 1]
                    prev_action = traj.actions[t - 1]
                    prev_forward_logits = self.forward_logits(prev_state, pref)
                    log_inflow = F.log_softmax(prev_forward_logits, dim=-1)[prev_action]

                # Outflow from state
                log_outflow = log_forward_prob

                # Flow matching condition
                fm_loss = (log_inflow - log_outflow) ** 2
                traj_loss = traj_loss + fm_loss

            losses.append(traj_loss)

        return torch.stack(losses).mean()

    def entropy_regularization(self,
                              trajectories: List[Trajectory],
                              preferences: List[torch.Tensor],
                              beta_reg: float = 0.01) -> torch.Tensor:
        """
        Compute entropy regularization term.

        Args:
            trajectories: List of trajectories
            preferences: List of preference vectors
            beta_reg: Regularization strength

        Returns:
            entropy_reg: Negative entropy (to be minimized)
        """
        entropies = []

        for traj, pref in zip(trajectories, preferences):
            traj_entropy = 0.0

            for state in traj.states[:-1]:  # Exclude terminal state
                # Forward policy entropy
                forward_logits = self.forward_logits(state, pref)
                forward_probs = F.softmax(forward_logits, dim=-1)
                forward_entropy = -(forward_probs * F.log_softmax(forward_logits, dim=-1)).sum()

                traj_entropy = traj_entropy + forward_entropy

            entropies.append(traj_entropy)

        # Return negative entropy (we want to maximize entropy = minimize -entropy)
        return -beta_reg * torch.stack(entropies).mean()

    def kl_regularization(self,
                         trajectories: List[Trajectory],
                         preferences: List[torch.Tensor],
                         beta_reg: float = 0.01) -> torch.Tensor:
        """
        Compute KL divergence regularization (between forward and backward policies).

        Args:
            trajectories: List of trajectories
            preferences: List of preference vectors
            beta_reg: Regularization strength

        Returns:
            kl_reg: KL divergence
        """
        kl_divs = []

        for traj, pref in zip(trajectories, preferences):
            traj_kl = 0.0

            for state in traj.states[:-1]:  # Exclude terminal state
                # Forward and backward policies
                forward_logits = self.forward_logits(state, pref)
                backward_logits = self.backward_logits(state, pref)

                forward_probs = F.softmax(forward_logits, dim=-1)
                log_forward_probs = F.log_softmax(forward_logits, dim=-1)
                log_backward_probs = F.log_softmax(backward_logits, dim=-1)

                # KL(forward || backward)
                kl = (forward_probs * (log_forward_probs - log_backward_probs)).sum()
                traj_kl = traj_kl + kl

            kl_divs.append(traj_kl)

        return beta_reg * torch.stack(kl_divs).mean()

    def compute_loss(self,
                    trajectories: List[Trajectory],
                    preferences: List[torch.Tensor],
                    beta: float = 1.0,
                    loss_type: str = 'trajectory_balance',
                    loss_params: Optional[Dict] = None,
                    regularization: str = 'none',
                    regularization_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute training loss for MOGFN.

        Args:
            trajectories: List of trajectories
            preferences: List of preference vectors
            beta: Reward exponent
            loss_type: Loss type ('trajectory_balance', 'detailed_balance',
                                  'subtrajectory_balance', 'flow_matching')
            loss_params: Additional parameters for the loss function
            regularization: Regularization type ('none', 'entropy', 'kl')
            regularization_params: Parameters for regularization

        Returns:
            loss: Training loss
        """
        if loss_params is None:
            loss_params = {}
        if regularization_params is None:
            regularization_params = {}

        # Compute base loss
        if loss_type == 'trajectory_balance':
            base_loss = self.trajectory_balance_loss(
                trajectories, preferences, beta,
                log_reward_clip=loss_params.get('log_reward_clip', 10.0)
            )
        elif loss_type == 'detailed_balance':
            base_loss = self.detailed_balance_loss(
                trajectories, preferences, beta,
                log_reward_clip=loss_params.get('log_reward_clip', 10.0)
            )
        elif loss_type == 'subtrajectory_balance':
            base_loss = self.subtrajectory_balance_loss(
                trajectories, preferences, beta,
                lambda_=loss_params.get('lambda_', 0.9),
                log_reward_clip=loss_params.get('log_reward_clip', 10.0)
            )
        elif loss_type == 'flow_matching':
            base_loss = self.flow_matching_loss(
                trajectories, preferences, beta,
                log_reward_clip=loss_params.get('log_reward_clip', 10.0)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Add regularization
        if regularization == 'entropy':
            reg_term = self.entropy_regularization(
                trajectories, preferences,
                beta_reg=regularization_params.get('beta', 0.01)
            )
            return base_loss + reg_term
        elif regularization == 'kl':
            reg_term = self.kl_regularization(
                trajectories, preferences,
                beta_reg=regularization_params.get('beta', 0.01)
            )
            return base_loss + reg_term
        else:
            return base_loss


class MultiObjectiveEnvironment(GFlowNetEnvironment):
    """
    Extended environment interface for multi-objective problems.
    """
    
    @abstractmethod
    def compute_objectives(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute all objective values for terminal state.
        
        Returns:
            objectives: Tensor of shape (num_objectives,)
        """
        pass
    
    @property
    @abstractmethod
    def num_objectives(self) -> int:
        """Number of objectives."""
        pass
    
    def compute_reward(self, state: torch.Tensor) -> torch.Tensor:
        """Return multi-objective rewards (overrides base class)."""
        return self.compute_objectives(state)


class PreferenceSampler:
    """
    Samples preference vectors from various distributions.
    """
    
    def __init__(self, 
                num_objectives: int,
                distribution: str = 'dirichlet',
                alpha: float = 1.5):
        """
        Args:
            num_objectives: Number of objectives
            distribution: 'dirichlet', 'uniform', or 'grid'
            alpha: Dirichlet concentration parameter
        """
        self.num_objectives = num_objectives
        self.distribution = distribution
        self.alpha = alpha
    
    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample preference vectors.
        
        Args:
            batch_size: Number of preferences to sample
        
        Returns:
            preferences: Tensor of shape (batch_size, num_objectives)
        """
        if self.distribution == 'dirichlet':
            alpha = np.ones(self.num_objectives) * self.alpha
            samples = np.random.dirichlet(alpha, size=batch_size)
            return torch.from_numpy(samples).float()
        
        elif self.distribution == 'uniform':
            # Uniform on simplex
            samples = np.random.exponential(size=(batch_size, self.num_objectives))
            samples = samples / samples.sum(axis=1, keepdims=True)
            return torch.from_numpy(samples).float()
        
        elif self.distribution == 'grid':
            # Uniform grid (only for small num_objectives)
            if self.num_objectives > 3:
                raise ValueError("Grid sampling only supported for ≤3 objectives")
            
            n_points = int(np.power(batch_size, 1/self.num_objectives)) + 1
            grid = np.linspace(0, 1, n_points)
            
            if self.num_objectives == 2:
                prefs = np.array([[1-x, x] for x in grid])
            elif self.num_objectives == 3:
                prefs = []
                for x in grid:
                    for y in grid:
                        z = 1 - x - y
                        if z >= 0:
                            prefs.append([x, y, z])
                prefs = np.array(prefs)
            
            # Randomly sample from grid if batch_size < grid size
            if len(prefs) > batch_size:
                indices = np.random.choice(len(prefs), batch_size, replace=False)
                prefs = prefs[indices]
            
            return torch.from_numpy(prefs).float()
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class MOGFNSampler:
    """Sampler for multi-objective GFlowNet."""

    def __init__(self,
                mogfn: MOGFN_PC,
                env: MultiObjectiveEnvironment,
                preference_sampler: PreferenceSampler,
                off_policy_ratio: float = 0.0):
        """
        Args:
            mogfn: MOGFN-PC model
            env: Multi-objective environment
            preference_sampler: Preference sampler
            off_policy_ratio: Probability of sampling random actions (off-policy)
        """
        self.mogfn = mogfn
        self.env = env
        self.preference_sampler = preference_sampler
        self.off_policy_ratio = off_policy_ratio
    
    def sample_trajectory(self, 
                        preference: torch.Tensor,
                        explore: bool = True) -> Trajectory:
        """
        Sample trajectory conditioned on preference.
        
        Args:
            preference: Preference vector
            explore: Whether to use exploration
        
        Returns:
            trajectory: Complete trajectory with multi-objective rewards
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

            # Off-policy exploration: with probability off_policy_ratio, sample uniformly random action
            if self.off_policy_ratio > 0 and torch.rand(1).item() < self.off_policy_ratio:
                # Sample uniform random action from valid actions
                action = valid_actions[torch.randint(len(valid_actions), (1,)).item()]
                # Compute log probability for this action under current policy
                logits = self.mogfn.forward_logits(state, preference)
                log_prob = torch.nn.functional.log_softmax(logits / self.mogfn.temperature, dim=-1)[action]
            else:
                # Sample action conditioned on preference (on-policy)
                action, log_prob = self.mogfn.sample_action(
                    state, preference, valid_actions, explore=explore
                )
            
            # Take step
            next_state, is_terminal = self.env.step(state, action)
            
            # Record
            actions.append(action)
            log_probs.append(log_prob)
            states.append(next_state)
            
            state = next_state
        
        # Compute multi-objective rewards
        objectives = self.env.compute_objectives(state)
        
        return Trajectory(
            states=states,
            actions=actions,
            log_probs=log_probs,
            is_terminal=True,
            reward=objectives  # Store multi-objective rewards
        )
    
    def sample_batch(self,
                    batch_size: int,
                    explore: bool = True,
                    fixed_preferences: Optional[torch.Tensor] = None) -> Tuple[List[Trajectory], List[torch.Tensor]]:
        """
        Sample batch of trajectories with preferences.
        
        Args:
            batch_size: Number of trajectories
            explore: Whether to explore
            fixed_preferences: Use these preferences instead of sampling (optional)
        
        Returns:
            trajectories: List of trajectories
            preferences: List of preference vectors used
        """
        if fixed_preferences is not None:
            preferences = fixed_preferences
            if len(preferences) != batch_size:
                raise ValueError(f"Fixed preferences length {len(preferences)} != batch_size {batch_size}")
        else:
            preferences = self.preference_sampler.sample(batch_size)
        
        trajectories = []
        for i in range(batch_size):
            traj = self.sample_trajectory(preferences[i], explore=explore)
            trajectories.append(traj)
        
        return trajectories, [preferences[i] for i in range(batch_size)]


class MOGFNTrainer:
    """Trainer for Multi-Objective GFlowNet."""

    def __init__(self,
                mogfn: MOGFN_PC,
                env: MultiObjectiveEnvironment,
                preference_sampler: PreferenceSampler,
                optimizer: torch.optim.Optimizer,
                beta: float = 1.0,
                off_policy_ratio: float = 0.0,
                loss_function: str = 'trajectory_balance',
                loss_params: Optional[Dict] = None,
                regularization: str = 'none',
                regularization_params: Optional[Dict] = None,
                modifications: str = 'none',
                modifications_params: Optional[Dict] = None,
                gradient_clip: float = 1.0):
        """
        Args:
            mogfn: MOGFN-PC model
            env: Multi-objective environment
            preference_sampler: Preference sampler
            optimizer: PyTorch optimizer
            beta: Reward exponent for scalarization
            off_policy_ratio: Probability of sampling random actions (off-policy)
            loss_function: Loss function type ('trajectory_balance', 'detailed_balance',
                                             'subtrajectory_balance', 'flow_matching')
            loss_params: Additional parameters for the loss function
            regularization: Regularization type ('none', 'entropy', 'kl')
            regularization_params: Parameters for regularization
            modifications: Modification type ('none', 'temperature_scaling', 'reward_shaping')
            modifications_params: Parameters for modifications
            gradient_clip: Maximum gradient norm for clipping
        """
        self.mogfn = mogfn
        self.env = env
        self.sampler = MOGFNSampler(mogfn, env, preference_sampler, off_policy_ratio=off_policy_ratio)
        self.optimizer = optimizer
        self.beta = beta
        self.loss_function = loss_function
        self.loss_params = loss_params or {}
        self.regularization = regularization
        self.regularization_params = regularization_params or {}
        self.modifications = modifications
        self.modifications_params = modifications_params or {}
        self.gradient_clip = gradient_clip

        self.losses = []
        self.iteration = 0

        # Track visited states for reward shaping (novelty bonus)
        self.visited_states = set()

    def apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Logits tensor

        Returns:
            scaled_logits: Temperature-scaled logits
        """
        if self.modifications != 'temperature_scaling':
            return logits

        temperature = self.modifications_params.get('temperature', 1.0)
        return logits / temperature

    def compute_novelty_bonus(self, state: torch.Tensor) -> float:
        """
        Compute novelty bonus for a state.

        Returns higher bonus for states that haven't been visited frequently.

        Args:
            state: State tensor

        Returns:
            novelty_bonus: Bonus reward for novelty (0 to 1)
        """
        from src.utils.tensor_utils import to_hashable

        # Convert state to hashable key
        state_key = to_hashable(state)

        # Count how many times we've seen this state
        if state_key in self.visited_states:
            # Already visited - no bonus
            return 0.0
        else:
            # Novel state - give bonus
            return 1.0

    def apply_reward_shaping(self, reward: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Apply reward shaping with diversity bonus.

        Args:
            reward: Original reward
            state: Current state

        Returns:
            shaped_reward: Reward with diversity bonus added
        """
        if self.modifications != 'reward_shaping':
            return reward

        gamma = self.modifications_params.get('gamma', 0.1)

        # Compute novelty bonus
        novelty_bonus = self.compute_novelty_bonus(state)

        # Add diversity bonus
        shaped_reward = reward + gamma * novelty_bonus

        # Track this state as visited
        from src.utils.tensor_utils import to_hashable
        state_key = to_hashable(state)
        self.visited_states.add(state_key)

        return shaped_reward

    def _apply_reward_shaping_to_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        Apply reward shaping to all trajectories.

        Args:
            trajectories: List of trajectories

        Returns:
            modified_trajectories: Trajectories with shaped rewards
        """
        gamma = self.modifications_params.get('gamma', 0.1)
        modified_trajectories = []

        for traj in trajectories:
            # Get the terminal state (last state)
            terminal_state = traj.states[-1]

            # Compute novelty bonus for this terminal state
            novelty_bonus = self.compute_novelty_bonus(terminal_state)

            # Shape the reward (multi-objective)
            original_reward = traj.reward
            shaped_reward = original_reward + gamma * novelty_bonus

            # Create new trajectory with shaped reward
            modified_traj = Trajectory(
                states=traj.states,
                actions=traj.actions,
                log_probs=traj.log_probs,
                is_terminal=traj.is_terminal,
                reward=shaped_reward
            )

            modified_trajectories.append(modified_traj)

            # Track this state as visited
            from src.utils.tensor_utils import to_hashable
            state_key = to_hashable(terminal_state)
            self.visited_states.add(state_key)

        return modified_trajectories

    def train_step(self, batch_size: int = 128, num_preferences_per_batch: Optional[int] = None) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch_size: Number of trajectories to sample
            num_preferences_per_batch: Number of preferences to sample (for batched sampling)

        Returns:
            metrics: Dictionary of training metrics
        """
        self.mogfn.train()

        # Apply temperature scaling modification if enabled
        original_temperature = None
        if self.modifications == 'temperature_scaling':
            temperature = self.modifications_params.get('temperature', 1.0)
            original_temperature = self.mogfn.temperature
            self.mogfn.temperature = temperature

        # Sample trajectories with preferences
        trajectories, preferences = self.sampler.sample_batch(batch_size, explore=True)

        # Restore original temperature if modified
        if original_temperature is not None:
            self.mogfn.temperature = original_temperature

        # Apply reward shaping modification if enabled
        if self.modifications == 'reward_shaping':
            trajectories = self._apply_reward_shaping_to_trajectories(trajectories)

        # Compute base loss (without regularization) for tracking
        base_loss = self.mogfn.compute_loss(
            trajectories,
            preferences,
            beta=self.beta,
            loss_type=self.loss_function,
            loss_params=self.loss_params,
            regularization='none',
            regularization_params={}
        )

        # Compute total loss (with regularization) for optimization
        total_loss = self.mogfn.compute_loss(
            trajectories,
            preferences,
            beta=self.beta,
            loss_type=self.loss_function,
            loss_params=self.loss_params,
            regularization=self.regularization,
            regularization_params=self.regularization_params
        )

        # Compute regularization term separately for tracking (detached)
        reg_term = total_loss.item() - base_loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.mogfn.parameters(), max_norm=self.gradient_clip)

        self.optimizer.step()

        # Track metrics
        total_loss_val = total_loss.item()
        base_loss_val = base_loss.item()

        self.losses.append(total_loss_val)
        self.iteration += 1

        metrics = {
            'loss': total_loss_val,
            'base_loss': base_loss_val,
            'reg_term': reg_term,
            'log_Z': self.mogfn.log_Z.item(),
            'iteration': self.iteration
        }

        return metrics
    
    def train(self,
            num_iterations: int,
            batch_size: int = 128,
            num_preferences_per_batch: Optional[int] = None,
            log_every: int = 100) -> Dict[str, List]:
        """
        Train for multiple iterations.

        Args:
            num_iterations: Number of training iterations
            batch_size: Batch size
            num_preferences_per_batch: Number of preferences to sample per batch
            log_every: Log frequency

        Returns:
            history: Training history
        """
        history = {
            'loss': [],
            'base_loss': [],
            'reg_term': [],
            'log_Z': [],
            'iteration': []
        }

        for i in range(num_iterations):
            metrics = self.train_step(batch_size, num_preferences_per_batch)

            if i % log_every == 0:
                print(f"Iteration {i}/{num_iterations} - Total Loss: {metrics['loss']:.4f}, "
                      f"Base Loss: {metrics['base_loss']:.4f}, Reg: {metrics['reg_term']:.4f}, "
                      f"log Z: {metrics['log_Z']:.4f}")

                for key, value in metrics.items():
                    if key in history:
                        history[key].append(value)

        return history
    
    def evaluate(self, 
                num_samples: int = 1000,
                preference: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """
        Evaluate the model.
        
        Args:
            num_samples: Number of samples to generate
            preference: Fixed preference (None = sample diverse preferences)
        
        Returns:
            results: Evaluation results
        """
        self.mogfn.eval()
        
        all_objectives = []
        all_preferences = []
        
        with torch.no_grad():
            if preference is not None:
                # Fixed preference
                preferences = preference.unsqueeze(0).repeat(num_samples, 1)
            else:
                # Sample diverse preferences
                preferences = self.sampler.preference_sampler.sample(num_samples)
            
            for i in range(num_samples):
                traj = self.sampler.sample_trajectory(preferences[i], explore=False)
                all_objectives.append(traj.reward)
                all_preferences.append(preferences[i])
        
        all_objectives = torch.stack(all_objectives)
        all_preferences = torch.stack(all_preferences)
        
        results = {
            'objectives': all_objectives,
            'preferences': all_preferences,
            'num_samples': num_samples
        }
        
        return results


def test_mogfn():
    """Test MOGFN implementation with a simple multi-objective environment."""
    
    class DummyMOEnvironment(MultiObjectiveEnvironment):
        """Simple 2-objective environment for testing."""
        
        def __init__(self):
            self._num_objectives = 2
            self._state_dim = 4
            self._num_actions = 4
        
        def get_initial_state(self):
            return torch.zeros(self._state_dim)
        
        def step(self, state, action):
            next_state = state.clone()
            next_state[action] = 1.0
            is_terminal = torch.sum(next_state) >= 3
            return next_state, is_terminal
        
        def get_valid_actions(self, state):
            return [i for i in range(self._num_actions) if state[i] == 0]
        
        def compute_objectives(self, state):
            # Two conflicting objectives
            obj1 = torch.sum(state[:2])  # Maximize first half
            obj2 = torch.sum(state[2:])  # Maximize second half
            return torch.tensor([obj1, obj2])
        
        @property
        def state_dim(self):
            return self._state_dim
        
        @property
        def num_actions(self):
            return self._num_actions
        
        @property
        def num_objectives(self):
            return self._num_objectives
    
    # Create environment
    env = DummyMOEnvironment()
    
    # Create MOGFN
    mogfn = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=64,
        num_actions=env.num_actions,
        num_layers=3,
        preference_encoding='vanilla',
        conditioning_type='concat'
    )
    
    # Create preference sampler
    pref_sampler = PreferenceSampler(
        num_objectives=env.num_objectives,
        distribution='dirichlet',
        alpha=1.5
    )
    
    # Create trainer
    optimizer = torch.optim.Adam(mogfn.parameters(), lr=1e-3)
    trainer = MOGFNTrainer(
        mogfn=mogfn,
        env=env,
        preference_sampler=pref_sampler,
        optimizer=optimizer,
        beta=1.0
    )
    
    # Train for a few iterations
    print("Training MOGFN-PC...")
    history = trainer.train(num_iterations=100, batch_size=16, log_every=20)
    
    # Evaluate
    print("\nEvaluating...")
    results = trainer.evaluate(num_samples=50)
    
    print(f"\nTest completed!")
    print(f"Generated {results['num_samples']} samples")
    print(f"Objective space coverage:")
    print(f"  Obj 1 - min: {results['objectives'][:, 0].min():.2f}, max: {results['objectives'][:, 0].max():.2f}")
    print(f"  Obj 2 - min: {results['objectives'][:, 1].min():.2f}, max: {results['objectives'][:, 1].max():.2f}")
    
    # Test different conditioning types
    print("\n" + "="*50)
    print("Testing FiLM conditioning...")
    mogfn_film = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=64,
        num_actions=env.num_actions,
        conditioning_type='film'
    )
    
    optimizer_film = torch.optim.Adam(mogfn_film.parameters(), lr=1e-3)
    trainer_film = MOGFNTrainer(mogfn_film, env, pref_sampler, optimizer_film)
    
    print("Training with FiLM conditioning...")
    history_film = trainer_film.train(num_iterations=100, batch_size=16, log_every=20)
    
    print("\nFiLM test completed!")
    
    # Test thermometer encoding
    print("\n" + "="*50)
    print("Testing thermometer encoding...")
    mogfn_therm = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=64,
        num_actions=env.num_actions,
        preference_encoding='thermometer',
        conditioning_type='concat'
    )
    
    optimizer_therm = torch.optim.Adam(mogfn_therm.parameters(), lr=1e-3)
    trainer_therm = MOGFNTrainer(mogfn_therm, env, pref_sampler, optimizer_therm)
    
    print("Training with thermometer encoding...")
    history_therm = trainer_therm.train(num_iterations=100, batch_size=16, log_every=20)
    
    print("\nThermometer encoding test completed!")
    print("\n" + "="*50)
    print("All tests passed successfully!")


if __name__ == '__main__':
    test_mogfn()
