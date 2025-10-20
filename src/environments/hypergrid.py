"""
HyperGrid environment for multi-objective GFlowNets.

Based on the environment described in:
    Jain et al. "Multi-Objective GFlowNets" (ICML 2023)

The HyperGrid is an H x H grid with multiple objectives defined as
rewards at different corners/regions of the grid.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.mogfn_pc import MultiObjectiveEnvironment


class HyperGrid(MultiObjectiveEnvironment):
    """
    H x H grid environment with multi-objective rewards.
    
    State: (x, y) position in grid, represented as one-hot or coordinate
    Actions: 0=right, 1=up, 2=done (terminate)
    
    The agent starts at (0,0) and can move right or up until reaching
    a terminal state by taking the 'done' action.
    
    Multiple objectives are defined by different reward functions
    at different regions of the grid.
    """
    
    def __init__(self,
                height: int = 8,
                num_objectives: int = 2,
                reward_config: str = 'corners',
                reward_beta: float = 1.0):
        """
        Args:
            height: Grid size (H x H)
            num_objectives: Number of objectives (typically 2-4)
            reward_config: 'corners', 'edges', or 'regions'
            reward_beta: Reward shaping parameter
        """
        self.height = height
        self._num_objectives = num_objectives
        self.reward_config = reward_config
        self.reward_beta = reward_beta
        
        # State representation: [x, y] coordinates
        self._state_dim = 2
        
        # Actions: 0=right, 1=up, 2=terminate
        self._num_actions = 3
        self.ACTION_RIGHT = 0
        self.ACTION_UP = 1
        self.ACTION_DONE = 2
        
        # Define reward functions for each objective
        self._setup_reward_functions()
    
    def _setup_reward_functions(self):
        """Setup reward functions based on configuration."""
        
        if self.reward_config == 'corners':
            # Standard setup from paper: rewards at 4 corners
            if self._num_objectives == 2:
                # R1: high at top-right, R2: high at top-left
                self.reward_functions = [
                    self._reward_top_right,
                    self._reward_top_left
                ]
            elif self._num_objectives == 3:
                # R1: top-right, R2: top-left, R3: bottom-right
                self.reward_functions = [
                    self._reward_top_right,
                    self._reward_top_left,
                    self._reward_bottom_right
                ]
            elif self._num_objectives == 4:
                # All four corners
                self.reward_functions = [
                    self._reward_top_right,
                    self._reward_top_left,
                    self._reward_bottom_right,
                    self._reward_bottom_left
                ]
            else:
                raise ValueError(f"Corners config only supports 2-4 objectives")
        
        elif self.reward_config == 'modes':
            # Multiple modes across the grid (from paper experiments)
            if self._num_objectives != 2:
                raise ValueError("Modes config only supports 2 objectives")
            
            self.reward_functions = [
                self._reward_modes_obj1,
                self._reward_modes_obj2
            ]
        
        else:
            raise ValueError(f"Unknown reward config: {self.reward_config}")
    
    def _reward_top_right(self, x: float, y: float) -> float:
        """Reward high at top-right corner (H-1, H-1)."""
        dist = np.sqrt((x - (self.height - 1))**2 + (y - (self.height - 1))**2)
        return np.exp(-0.5 * dist)
    
    def _reward_top_left(self, x: float, y: float) -> float:
        """Reward high at top-left corner (0, H-1)."""
        dist = np.sqrt((x - 0)**2 + (y - (self.height - 1))**2)
        return np.exp(-0.5 * dist)
    
    def _reward_bottom_right(self, x: float, y: float) -> float:
        """Reward high at bottom-right corner (H-1, 0)."""
        dist = np.sqrt((x - (self.height - 1))**2 + (y - 0)**2)
        return np.exp(-0.5 * dist)
    
    def _reward_bottom_left(self, x: float, y: float) -> float:
        """Reward high at bottom-left corner (0, 0)."""
        dist = np.sqrt((x - 0)**2 + (y - 0)**2)
        return np.exp(-0.5 * dist)
    
    def _reward_modes_obj1(self, x: float, y: float) -> float:
        """Multiple modes for objective 1."""
        # Modes at multiple locations
        modes = [(2, 6), (6, 2), (4, 4)]
        rewards = [np.exp(-0.5 * ((x - mx)**2 + (y - my)**2)) for mx, my in modes]
        return max(rewards)
    
    def _reward_modes_obj2(self, x: float, y: float) -> float:
        """Multiple modes for objective 2."""
        modes = [(1, 2), (6, 6), (2, 4)]
        rewards = [np.exp(-0.5 * ((x - mx)**2 + (y - my)**2)) for mx, my in modes]
        return max(rewards)
    
    def get_initial_state(self) -> torch.Tensor:
        """Get initial state at (0, 0)."""
        return torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    def step(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, bool]:
        """
        Take action in environment.
        
        Args:
            state: Current state [x, y]
            action: 0=right, 1=up, 2=done
        
        Returns:
            next_state: Next state
            is_terminal: Whether episode is done
        """
        x, y = state[0].item(), state[1].item()
        
        if action == self.ACTION_RIGHT:
            # Move right (increase x)
            x = min(x + 1, self.height - 1)
            is_terminal = False
        
        elif action == self.ACTION_UP:
            # Move up (increase y)
            y = min(y + 1, self.height - 1)
            is_terminal = False
        
        elif action == self.ACTION_DONE:
            # Terminate
            is_terminal = True
        
        else:
            raise ValueError(f"Invalid action: {action}")
        
        next_state = torch.tensor([x, y], dtype=torch.float32)
        return next_state, is_terminal
    
    def get_valid_actions(self, state: torch.Tensor) -> List[int]:
        """
        Get valid actions for current state.
        
        Always can terminate. Can move right/up if not at boundary.
        """
        x, y = state[0].item(), state[1].item()
        valid_actions = [self.ACTION_DONE]  # Can always terminate
        
        if x < self.height - 1:
            valid_actions.append(self.ACTION_RIGHT)
        
        if y < self.height - 1:
            valid_actions.append(self.ACTION_UP)
        
        return valid_actions
    
    def compute_objectives(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute all objective values for state.
        
        Args:
            state: Terminal state [x, y]
        
        Returns:
            objectives: Tensor of shape (num_objectives,)
        """
        x, y = state[0].item(), state[1].item()
        
        objectives = []
        for reward_fn in self.reward_functions:
            reward = reward_fn(x, y)
            objectives.append(reward)
        
        return torch.tensor(objectives, dtype=torch.float32)
    
    @property
    def state_dim(self) -> int:
        return self._state_dim
    
    @property
    def num_actions(self) -> int:
        return self._num_actions
    
    @property
    def num_objectives(self) -> int:
        return self._num_objectives
    
    def visualize_objectives(self, save_path: Optional[str] = None):
        """
        Visualize the reward functions across the grid.
        
        Creates a plot showing each objective's reward landscape.
        """
        import matplotlib.pyplot as plt
        
        # Create grid of positions
        x_coords = np.arange(self.height)
        y_coords = np.arange(self.height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Compute rewards for each objective
        fig, axes = plt.subplots(1, self._num_objectives, figsize=(5*self._num_objectives, 4))
        
        if self._num_objectives == 1:
            axes = [axes]
        
        for i, (ax, reward_fn) in enumerate(zip(axes, self.reward_functions)):
            Z = np.zeros_like(X, dtype=float)
            
            for ix in range(self.height):
                for iy in range(self.height):
                    Z[iy, ix] = reward_fn(ix, iy)
            
            im = ax.imshow(Z, origin='lower', cmap='viridis', aspect='auto')
            ax.set_title(f'Objective {i+1}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved objective visualization to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def visualize_samples(self, 
                        objectives: torch.Tensor,
                        preferences: Optional[torch.Tensor] = None,
                        save_path: Optional[str] = None):
        """
        Visualize sampled solutions in objective space.
        
        Args:
            objectives: Tensor of shape (N, num_objectives)
            preferences: Optional preference vectors
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        objectives_np = objectives.detach().cpu().numpy()
        
        if self._num_objectives == 2:
            # 2D scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(objectives_np[:, 0], objectives_np[:, 1], 
                    alpha=0.6, s=30, c='blue')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title('Sampled Solutions in Objective Space')
            plt.grid(True, alpha=0.3)
            
        elif self._num_objectives == 3:
            # 3D scatter plot
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives_np[:, 0], objectives_np[:, 1], objectives_np[:, 2],
                    alpha=0.6, s=30, c='blue')
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title('Sampled Solutions in Objective Space')
        
        else:
            # Parallel coordinates for >3 objectives
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(len(objectives_np)):
                ax.plot(range(self._num_objectives), objectives_np[i], 
                    alpha=0.3, c='blue')
            ax.set_xlabel('Objective Index')
            ax.set_ylabel('Objective Value')
            ax.set_title('Sampled Solutions (Parallel Coordinates)')
            ax.set_xticks(range(self._num_objectives))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved samples visualization to {save_path}")
        else:
            plt.show()


def test_hypergrid():
    """Test HyperGrid environment."""
    
    print("Testing HyperGrid Environment")
    print("=" * 50)
    
    # Create environment
    env = HyperGrid(height=8, num_objectives=2, reward_config='corners')
    
    print(f"Grid size: {env.height} x {env.height}")
    print(f"State dim: {env.state_dim}")
    print(f"Num actions: {env.num_actions}")
    print(f"Num objectives: {env.num_objectives}")
    
    # Test initial state
    state = env.get_initial_state()
    print(f"\nInitial state: {state}")
    
    # Test actions
    print("\nTesting actions from (0,0):")
    valid_actions = env.get_valid_actions(state)
    print(f"Valid actions: {valid_actions}")
    
    # Move right
    state, done = env.step(state, env.ACTION_RIGHT)
    print(f"After moving right: {state}, done={done}")
    
    # Move up
    state, done = env.step(state, env.ACTION_UP)
    print(f"After moving up: {state}, done={done}")
    
    # Test objectives at different positions
    print("\nObjective values at different positions:")
    test_positions = [
        (0, 0),  # Bottom-left
        (7, 7),  # Top-right
        (0, 7),  # Top-left
        (7, 0),  # Bottom-right
        (4, 4),  # Center
    ]
    
    for x, y in test_positions:
        state = torch.tensor([float(x), float(y)])
        objectives = env.compute_objectives(state)
        print(f"  Position ({x}, {y}): {objectives.numpy()}")
    
    # Visualize objectives
    print("\nGenerating objective landscape visualization...")
    try:
        env.visualize_objectives(save_path='hypergrid_objectives.png')
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    
    print("\nHyperGrid test completed successfully!")


if __name__ == '__main__':
    test_hypergrid()