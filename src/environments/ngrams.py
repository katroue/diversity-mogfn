"""
N-grams environment for multi-objective GFlowNets.

Based on the environment described in:
    Jain et al. "Multi-Objective GFlowNets" (ICML 2023)

The N-grams task involves generating sequences of fixed length from a vocabulary,
where multiple objectives are defined as counts of different n-gram patterns.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.mogfn_pc import MultiObjectiveEnvironment


class NGrams(MultiObjectiveEnvironment):
    """
    N-grams sequence generation environment with multi-objective rewards.

    State: Partially constructed sequence (list of character indices)
    Actions: Append character from vocabulary (0 to vocab_size-1) or DONE

    The agent starts with an empty sequence and adds one character at a time
    until reaching the target length, then terminates.

    Multiple objectives are defined by counting occurrences of different
    n-gram patterns (e.g., "AA", "BB", "AB", etc.).
    """

    def __init__(self,
                 vocab_size: int = 4,
                 seq_length: int = 8,
                 ngram_length: int = 2,
                 objective_patterns: Optional[List[str]] = None,
                 normalize_rewards: bool = True):
        """
        Args:
            vocab_size: Size of vocabulary (e.g., 4 for A,B,C,D)
            seq_length: Length of sequences to generate
            ngram_length: Length of n-grams to count (2 for bigrams, 3 for trigrams)
            objective_patterns: List of n-gram patterns to use as objectives
                               If None, uses default patterns based on vocab_size
            normalize_rewards: Whether to normalize objective counts by max possible
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.ngram_length = ngram_length
        self.normalize_rewards = normalize_rewards

        # Create vocabulary (A, B, C, ...)
        self.vocab = [chr(ord('A') + i) for i in range(vocab_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}

        # Setup objective patterns
        if objective_patterns is None:
            objective_patterns = self._default_objective_patterns()

        self.objective_patterns = objective_patterns
        self._num_objectives = len(objective_patterns)

        # State representation: sequence of character indices + position
        # We use a fixed-size representation: [pos, char_0, char_1, ..., char_{L-1}]
        # where pos is current position and char_i is -1 if not filled yet
        self._state_dim = 1 + seq_length  # position + sequence

        # Actions: 0 to vocab_size-1 (append character), vocab_size (DONE)
        self._num_actions = vocab_size + 1
        self.ACTION_DONE = vocab_size

        # Maximum possible count for each n-gram (for normalization)
        self.max_count = seq_length - ngram_length + 1

    def _default_objective_patterns(self) -> List[str]:
        """
        Generate default objective patterns based on vocabulary.

        For vocab_size=4 (A,B,C,D) and ngram_length=2:
        - Use patterns: AA, BB, AB, BA (4 objectives)

        Returns:
            List of n-gram pattern strings
        """
        if self.ngram_length == 2:
            if self.vocab_size >= 2:
                # Use homogeneous bigrams (AA, BB) and heterogeneous (AB, BA)
                patterns = [
                    self.vocab[0] * 2,  # AA
                    self.vocab[1] * 2,  # BB
                ]
                if self.vocab_size >= 2:
                    patterns.extend([
                        self.vocab[0] + self.vocab[1],  # AB
                        self.vocab[1] + self.vocab[0],  # BA
                    ])
                return patterns
            else:
                return [self.vocab[0] * 2]  # Just AA

        elif self.ngram_length == 3:
            # For trigrams, use some common patterns
            if self.vocab_size >= 2:
                return [
                    self.vocab[0] * 3,  # AAA
                    self.vocab[1] * 3,  # BBB
                    self.vocab[0] * 2 + self.vocab[1],  # AAB
                    self.vocab[1] * 2 + self.vocab[0],  # BBA
                ]
            else:
                return [self.vocab[0] * 3]

        else:
            # General case: use first few homogeneous n-grams
            return [self.vocab[i] * self.ngram_length
                   for i in range(min(self.vocab_size, 4))]

    def get_initial_state(self) -> torch.Tensor:
        """
        Get initial state (empty sequence).

        Returns:
            state: [position=0, -1, -1, ..., -1] (all chars unfilled)
        """
        state = torch.full((self._state_dim,), -1.0, dtype=torch.float32)
        state[0] = 0.0  # Position starts at 0
        return state

    def step(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, bool]:
        """
        Take action in environment (append character to sequence).

        Args:
            state: Current state [pos, char_0, ..., char_{L-1}]
            action: Character index to append (0 to vocab_size-1) or DONE

        Returns:
            next_state: Next state
            is_terminal: Whether episode is done
        """
        # Clone state to avoid modifying original
        next_state = state.clone()

        pos = int(state[0].item())

        if action == self.ACTION_DONE:
            # Terminal action - sequence must be complete
            is_terminal = True

        elif action < self.vocab_size:
            # Append character at current position
            if pos < self.seq_length:
                next_state[1 + pos] = float(action)
                next_state[0] = float(pos + 1)  # Increment position
            is_terminal = (pos + 1) >= self.seq_length

        else:
            raise ValueError(f"Invalid action: {action}")

        return next_state, is_terminal

    def get_valid_actions(self, state: torch.Tensor) -> List[int]:
        """
        Get valid actions for current state.

        If sequence is incomplete: can append any character
        If sequence is complete: must take DONE action
        """
        pos = int(state[0].item())

        if pos >= self.seq_length:
            # Sequence complete - must terminate
            return [self.ACTION_DONE]
        else:
            # Can append any character
            return list(range(self.vocab_size))

    def _state_to_sequence(self, state: torch.Tensor) -> str:
        """
        Convert state tensor to sequence string.

        Args:
            state: State tensor [pos, char_0, ..., char_{L-1}]

        Returns:
            sequence: String representation (e.g., "ABBA")
        """
        pos = int(state[0].item())
        chars = []

        for i in range(pos):
            char_idx = int(state[1 + i].item())
            if char_idx >= 0:
                chars.append(self.vocab[char_idx])

        return ''.join(chars)

    def _count_ngrams(self, sequence: str, pattern: str) -> int:
        """
        Count occurrences of pattern in sequence.

        Args:
            sequence: Sequence string (e.g., "ABBA")
            pattern: N-gram pattern to count (e.g., "AB")

        Returns:
            count: Number of occurrences
        """
        count = 0
        pattern_len = len(pattern)

        for i in range(len(sequence) - pattern_len + 1):
            if sequence[i:i + pattern_len] == pattern:
                count += 1

        return count

    def compute_objectives(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute all objective values for terminal state.

        Each objective is the count of a specific n-gram pattern.

        Args:
            state: Terminal state

        Returns:
            objectives: Tensor of shape (num_objectives,) with counts/frequencies
        """
        sequence = self._state_to_sequence(state)

        objectives = []
        for pattern in self.objective_patterns:
            count = self._count_ngrams(sequence, pattern)

            if self.normalize_rewards:
                # Normalize by maximum possible count
                objectives.append(count / self.max_count if self.max_count > 0 else 0.0)
            else:
                objectives.append(float(count))

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

    def get_pattern_description(self, idx: int) -> str:
        """Get description of objective pattern."""
        return f"Count of '{self.objective_patterns[idx]}'"

    def visualize_samples(self,
                         objectives: torch.Tensor,
                         sequences: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
        """
        Visualize sampled solutions in objective space.

        Args:
            objectives: Tensor of shape (N, num_objectives)
            sequences: Optional list of sequence strings
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt

        objectives_np = objectives.detach().cpu().numpy()

        if self._num_objectives == 2:
            # 2D scatter plot
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(objectives_np[:, 0], objectives_np[:, 1],
                                 alpha=0.6, s=50, c='blue')

            # Annotate some points with sequences if provided
            if sequences and len(sequences) <= 20:
                for i in range(min(len(sequences), 20)):
                    plt.annotate(sequences[i],
                               (objectives_np[i, 0], objectives_np[i, 1]),
                               fontsize=8, alpha=0.7,
                               xytext=(5, 5), textcoords='offset points')

            plt.xlabel(f'Objective 1: {self.get_pattern_description(0)}')
            plt.ylabel(f'Objective 2: {self.get_pattern_description(1)}')
            plt.title(f'N-grams Solutions (L={self.seq_length}, vocab={self.vocab_size})')
            plt.grid(True, alpha=0.3)

        elif self._num_objectives == 3:
            # 3D scatter plot
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives_np[:, 0], objectives_np[:, 1], objectives_np[:, 2],
                      alpha=0.6, s=50, c='blue')
            ax.set_xlabel(f'Obj 1: {self.objective_patterns[0]}')
            ax.set_ylabel(f'Obj 2: {self.objective_patterns[1]}')
            ax.set_zlabel(f'Obj 3: {self.objective_patterns[2]}')
            ax.set_title('N-grams Solutions (3D)')

        elif self._num_objectives == 4:
            # 2x2 grid of pairwise scatter plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
            for idx, (i, j) in enumerate(pairs):
                if i < self._num_objectives and j < self._num_objectives:
                    axes[idx].scatter(objectives_np[:, i], objectives_np[:, j],
                                    alpha=0.6, s=30, c='blue')
                    axes[idx].set_xlabel(self.objective_patterns[i])
                    axes[idx].set_ylabel(self.objective_patterns[j])
                    axes[idx].grid(True, alpha=0.3)

            plt.suptitle(f'N-grams Solutions (Pairwise)', fontsize=14)

        else:
            # Parallel coordinates for many objectives
            fig, ax = plt.subplots(figsize=(12, 6))
            for i in range(len(objectives_np)):
                ax.plot(range(self._num_objectives), objectives_np[i],
                       alpha=0.3, c='blue', linewidth=0.5)
            ax.set_xlabel('Objective Index')
            ax.set_ylabel('Objective Value (Normalized Count)')
            ax.set_title('N-grams Solutions (Parallel Coordinates)')
            ax.set_xticks(range(self._num_objectives))
            ax.set_xticklabels(self.objective_patterns, rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()


def test_ngrams():
    """Test NGrams environment."""

    print("Testing NGrams Environment")
    print("=" * 60)

    # Create environment
    env = NGrams(vocab_size=4, seq_length=8, ngram_length=2)

    print(f"Vocabulary: {env.vocab}")
    print(f"Sequence length: {env.seq_length}")
    print(f"N-gram length: {env.ngram_length}")
    print(f"State dim: {env.state_dim}")
    print(f"Num actions: {env.num_actions}")
    print(f"Num objectives: {env.num_objectives}")
    print(f"Objective patterns: {env.objective_patterns}")

    # Test initial state
    state = env.get_initial_state()
    print(f"\nInitial state: {state}")
    print(f"Sequence: '{env._state_to_sequence(state)}'")

    # Build a test sequence: "AABBAABB"
    print("\nBuilding sequence 'AABBAABB':")
    actions = [0, 0, 1, 1, 0, 0, 1, 1]  # A, A, B, B, A, A, B, B

    for i, action in enumerate(actions):
        valid_actions = env.get_valid_actions(state)
        print(f"  Step {i}: action={env.vocab[action]}, valid_actions={valid_actions}")

        state, done = env.step(state, action)
        sequence = env._state_to_sequence(state)
        print(f"    State: {state[:5]}..., sequence='{sequence}', done={done}")

    # Compute objectives for final state
    print(f"\nFinal sequence: '{env._state_to_sequence(state)}'")
    objectives = env.compute_objectives(state)
    print(f"Objectives: {objectives}")

    for i, pattern in enumerate(env.objective_patterns):
        print(f"  {pattern}: {objectives[i].item():.3f}")

    # Test different sequences
    print("\n" + "="*60)
    print("Testing different sequences:")
    print("="*60)

    test_sequences = [
        "AAAAAAAA",  # All A's
        "BBBBBBBB",  # All B's
        "ABABABAB",  # Alternating
        "AABBCCDD",  # Pairs
        "ABCDABCD",  # Repeated pattern
    ]

    for seq_str in test_sequences:
        # Build state for this sequence
        state = env.get_initial_state()
        for char in seq_str[:env.seq_length]:
            char_idx = env.char_to_idx.get(char, 0)
            state, _ = env.step(state, char_idx)

        objectives = env.compute_objectives(state)
        print(f"\n'{seq_str}':")
        for i, pattern in enumerate(env.objective_patterns):
            count = env._count_ngrams(seq_str, pattern)
            print(f"  {pattern}: count={count}, normalized={objectives[i].item():.3f}")

    # Test with different configurations
    print("\n" + "="*60)
    print("Testing different configurations:")
    print("="*60)

    # Trigrams
    env_trigrams = NGrams(vocab_size=2, seq_length=9, ngram_length=3)
    print(f"\nTrigrams (vocab=2, L=9):")
    print(f"  Patterns: {env_trigrams.objective_patterns}")

    # Larger vocabulary
    env_large_vocab = NGrams(vocab_size=8, seq_length=10, ngram_length=2)
    print(f"\nLarger vocabulary (vocab=8, L=10):")
    print(f"  Vocabulary: {env_large_vocab.vocab}")
    print(f"  Patterns: {env_large_vocab.objective_patterns}")

    print("\n" + "="*60)
    print("NGrams test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    test_ngrams()
