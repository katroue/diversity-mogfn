"""
DNA Sequence Generation Environment for Multi-Objective GFlowNets.

This environment generates DNA sequences character by character, with multiple
biological objectives including GC content, motif presence, entropy, and
homopolymer penalties.

Based on the Multi-Objective GFlowNets paper (Jain et al., ICML 2023).
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.mogfn_pc import MultiObjectiveEnvironment


class DNASequence(MultiObjectiveEnvironment):
    """
    DNA sequence generation environment.

    State: [position, base_0, base_1, ..., base_{L-1}]
        - position: current position in sequence (0 to seq_length)
        - base_i: nucleotide at position i (0=A, 1=C, 2=G, 3=T, -1=empty)

    Actions:
        - 0-3: Append base (0=A, 1=C, 2=G, 3=T)
        - 4: DONE (terminate sequence)

    Objectives (as specified in Jain et al., ICML 2023):
        - free_energy: Free energy of secondary structure (from NUPACK), lower is better (we maximize -energy)
        - num_base_pairs: Number of base pairs in secondary structure
        - inverse_length: 1/length (favors shorter sequences)

    Example:
        >>> env = DNASequence(seq_length=10)
        >>> state = env.get_initial_state()
        >>> state, done = env.step(state, 0)  # Append 'A'
        >>> state, done = env.step(state, 2)  # Append 'G'
        >>> objectives = env.compute_objectives(state)
        >>> # objectives = [free_energy, num_base_pairs, inverse_length]
    """

    # DNA alphabet
    BASES = ['A', 'C', 'G', 'T']
    BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __init__(self,
                 seq_length: int = 20,
                 objectives: Optional[List[str]] = None,
                 temperature: float = 37.0,
                 use_nupack: bool = True):
        """
        Initialize DNA sequence environment (paper specification).

        Args:
            seq_length: Maximum sequence length
            objectives: List of objective names to optimize
                Options: 'free_energy', 'num_base_pairs', 'inverse_length'
                Default: all three objectives
            temperature: Temperature in Celsius for NUPACK calculations (default 37°C)
            use_nupack: If True, try to use NUPACK for free energy calculation
                       If False or NUPACK unavailable, use simple heuristics
        """
        self.seq_length = seq_length
        self.temperature = temperature
        self.use_nupack = use_nupack

        # Try to import NUPACK
        self._nupack_available = False
        if use_nupack:
            try:
                import nupack
                self._nupack_available = True
                self._nupack = nupack
                print(f"✓ NUPACK is available for free energy calculations")
            except ImportError:
                print(f"⚠ NUPACK not available, using heuristic approximations")
                print(f"  Install with: pip install nupack")

        # Set objectives (paper specification: free energy, num base pairs, inverse length)
        if objectives is None:
            self.objectives = ['free_energy', 'num_base_pairs', 'inverse_length']
        else:
            self.objectives = objectives

        # Validate objectives
        valid_objectives = ['free_energy', 'num_base_pairs', 'inverse_length']
        for obj in self.objectives:
            if obj not in valid_objectives:
                raise ValueError(f"Invalid objective '{obj}'. Must be one of {valid_objectives}")

        # Environment properties
        self._num_objectives = len(self.objectives)
        self._state_dim = 1 + seq_length  # position + bases
        self._num_actions = 5  # 4 bases + DONE

        # DONE action index
        self.DONE_ACTION = 4

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def num_objectives(self) -> int:
        return self._num_objectives

    def get_initial_state(self) -> torch.Tensor:
        """
        Get initial empty state.

        Returns:
            state: [position=0, base_0=-1, ..., base_{L-1}=-1]
        """
        state = torch.zeros(self._state_dim)
        state[0] = 0  # position
        state[1:] = -1  # empty bases
        return state

    def step(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, bool]:
        """
        Take a step in the environment.

        Args:
            state: Current state
            action: Action to take (0-3 for bases, 4 for DONE)

        Returns:
            next_state: New state after action
            done: Whether sequence is complete
        """
        state = state.clone()
        position = int(state[0].item())

        # DONE action
        if action == self.DONE_ACTION:
            return state, True

        # Append base
        if position < self.seq_length:
            state[1 + position] = action
            state[0] += 1
            position += 1

        # Check if sequence is full
        done = (position >= self.seq_length)

        return state, done

    def get_valid_actions(self, state: torch.Tensor) -> List[int]:
        """
        Get valid actions for current state.

        Args:
            state: Current state

        Returns:
            valid_actions: List of valid action indices
        """
        position = int(state[0].item())

        # If at maximum length, only DONE is valid
        if position >= self.seq_length:
            return [self.DONE_ACTION]

        # Otherwise, can append any base or terminate
        return list(range(self._num_actions))

    def _state_to_sequence(self, state: torch.Tensor) -> str:
        """
        Convert state to DNA sequence string.

        Args:
            state: State tensor

        Returns:
            sequence: DNA sequence string (e.g., "ACGT")
        """
        position = int(state[0].item())
        bases = state[1:1+position].long().cpu().numpy()

        sequence = ''.join([self.BASES[b] for b in bases if 0 <= b < 4])
        return sequence

    def compute_objectives(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute objective values for a sequence (paper specification).

        Objectives:
            1. free_energy: -free_energy from NUPACK (maximizing stability)
            2. num_base_pairs: Number of base pairs in secondary structure
            3. inverse_length: 1/length (favoring shorter sequences)

        Args:
            state: State tensor

        Returns:
            objectives: Tensor of objective values, shape (num_objectives,)
        """
        sequence = self._state_to_sequence(state)

        if len(sequence) == 0:
            # Empty sequence gets zero reward
            return torch.zeros(self._num_objectives)

        objective_values = []

        for obj_name in self.objectives:
            if obj_name == 'free_energy':
                value = self._compute_free_energy(sequence)
            elif obj_name == 'num_base_pairs':
                value = self._compute_num_base_pairs(sequence)
            elif obj_name == 'inverse_length':
                value = self._compute_inverse_length(sequence)
            else:
                raise ValueError(f"Unknown objective: {obj_name}")

            objective_values.append(value)

        return torch.tensor(objective_values, dtype=torch.float32)

    def _compute_free_energy(self, sequence: str) -> float:
        """
        Compute free energy of secondary structure using NUPACK.

        Lower free energy = more stable structure = better.
        We return -free_energy so that maximization works.

        Args:
            sequence: DNA sequence string

        Returns:
            score: Negative free energy (higher is better/more stable)
                   Normalized to ~[0, 1] range for compatibility
        """
        if len(sequence) == 0:
            return 0.0

        if self._nupack_available:
            try:
                # Create NUPACK strand
                strand = self._nupack.Strand(sequence, name='seq')

                # Create complex
                complex_ = self._nupack.Complex([strand], name='complex')

                # Set model (DNA, temperature in Kelvin)
                model = self._nupack.Model(material='dna', celsius=self.temperature)

                # Compute MFE (minimum free energy) structure
                result = self._nupack.mfe([complex_], model=model)

                # Get free energy (in kcal/mol)
                free_energy = float(result[0].energy)

                # Return negative energy (so more negative/stable = higher reward)
                # Normalize: typical range is -30 to 0 kcal/mol
                # Map to [0, 1]: -30 kcal/mol → 1.0, 0 kcal/mol → 0.0
                normalized_score = min(1.0, max(0.0, -free_energy / 30.0))

                return float(normalized_score)

            except Exception as e:
                # Fall back to heuristic if NUPACK fails
                print(f"Warning: NUPACK calculation failed: {e}")
                return self._compute_free_energy_heuristic(sequence)
        else:
            # Use heuristic approximation
            return self._compute_free_energy_heuristic(sequence)

    def _compute_free_energy_heuristic(self, sequence: str) -> float:
        """
        Heuristic approximation of free energy (when NUPACK unavailable).

        Based on GC content and potential base pairing.
        GC pairs are more stable than AT pairs.

        Args:
            sequence: DNA sequence string

        Returns:
            score: Approximate free energy score (0 to 1)
        """
        if len(sequence) == 0:
            return 0.0

        # Simple heuristic: GC content correlates with stability
        gc_count = sequence.count('G') + sequence.count('C')
        gc_ratio = gc_count / len(sequence)

        # More GC = more stable (in general)
        # Also consider sequence length (longer = more potential pairs)
        length_factor = min(1.0, len(sequence) / self.seq_length)

        score = (gc_ratio * 0.7 + length_factor * 0.3)

        return float(score)

    def _compute_num_base_pairs(self, sequence: str) -> float:
        """
        Compute number of base pairs in secondary structure.

        More base pairs = more structure = better.

        Args:
            sequence: DNA sequence string

        Returns:
            score: Number of base pairs, normalized by sequence length
        """
        if len(sequence) == 0:
            return 0.0

        if self._nupack_available:
            try:
                # Create NUPACK strand
                strand = self._nupack.Strand(sequence, name='seq')

                # Create complex
                complex_ = self._nupack.Complex([strand], name='complex')

                # Set model
                model = self._nupack.Model(material='dna', celsius=self.temperature)

                # Compute MFE structure
                result = self._nupack.mfe([complex_], model=model)

                # Get structure (dot-bracket notation)
                structure = str(result[0].structure)

                # Count base pairs (opening and closing brackets)
                num_pairs = structure.count('(')  # Each '(' has a matching ')'

                # Normalize by maximum possible pairs (length / 2)
                max_pairs = len(sequence) // 2
                if max_pairs > 0:
                    normalized_score = num_pairs / max_pairs
                else:
                    normalized_score = 0.0

                return float(normalized_score)

            except Exception as e:
                print(f"Warning: NUPACK calculation failed: {e}")
                return self._compute_num_base_pairs_heuristic(sequence)
        else:
            # Use heuristic approximation
            return self._compute_num_base_pairs_heuristic(sequence)

    def _compute_num_base_pairs_heuristic(self, sequence: str) -> float:
        """
        Heuristic approximation of number of base pairs.

        Uses simple complementary base counting.

        Args:
            sequence: DNA sequence string

        Returns:
            score: Approximate number of base pairs, normalized
        """
        if len(sequence) == 0:
            return 0.0

        # Count complementary bases
        # A pairs with T, G pairs with C
        a_count = sequence.count('A')
        t_count = sequence.count('T')
        g_count = sequence.count('G')
        c_count = sequence.count('C')

        # Maximum potential pairs
        at_pairs = min(a_count, t_count)
        gc_pairs = min(g_count, c_count)
        total_pairs = at_pairs + gc_pairs

        # Normalize by maximum possible pairs
        max_pairs = len(sequence) // 2
        if max_pairs > 0:
            score = total_pairs / max_pairs
        else:
            score = 0.0

        return float(score)

    def _compute_inverse_length(self, sequence: str) -> float:
        """
        Compute inverse of sequence length.

        Shorter sequences = higher score (as per paper).

        Args:
            sequence: DNA sequence string

        Returns:
            score: 1/length, normalized by maximum length
        """
        if len(sequence) == 0:
            return 0.0

        # Inverse length, normalized
        # 1 / 1 = 1.0 (shortest)
        # 1 / seq_length = minimum
        # Normalize so that length=1 → 1.0, length=seq_length → ~0
        score = 1.0 / len(sequence)

        # Normalize by maximum possible (which is 1/1 = 1.0)
        # Already in [0, 1] range

        return float(score)

    def sample_random_trajectory(self) -> torch.Tensor:
        """
        Sample a random complete trajectory.

        Returns:
            state: Final state of random trajectory
        """
        state = self.get_initial_state()
        done = False

        while not done:
            valid_actions = self.get_valid_actions(state)

            # Random action (biased away from early termination)
            position = int(state[0].item())
            if position < self.seq_length * 0.5:
                # Early in sequence: only append bases
                valid_actions = [a for a in valid_actions if a != self.DONE_ACTION]

            if len(valid_actions) == 0:
                break

            action = np.random.choice(valid_actions)
            state, done = self.step(state, action)

        return state

    def get_sequence_length(self, state: torch.Tensor) -> int:
        """
        Get the length of the sequence in the current state.

        Args:
            state: State tensor

        Returns:
            length: Number of bases in sequence
        """
        return int(state[0].item())

    def get_base_name(self, base_idx: int) -> str:
        """
        Get base name from index.

        Args:
            base_idx: Base index (0-3)

        Returns:
            base: Base name ('A', 'C', 'G', or 'T')
        """
        if 0 <= base_idx < len(self.BASES):
            return self.BASES[base_idx]
        return '?'

    def visualize_sequence(self, state: torch.Tensor) -> str:
        """
        Create a visual representation of the sequence.

        Args:
            state: State tensor

        Returns:
            viz: Multi-line string visualization
        """
        sequence = self._state_to_sequence(state)
        objectives = self.compute_objectives(state)

        lines = []
        lines.append("DNA Sequence:")
        lines.append(f"  {sequence}")
        lines.append(f"  Length: {len(sequence)}")
        lines.append("")
        lines.append("Objectives:")

        for i, obj_name in enumerate(self.objectives):
            value = objectives[i].item()
            lines.append(f"  {obj_name}: {value:.4f}")

        # Add sequence analysis
        if len(sequence) > 0:
            lines.append("")
            lines.append("Analysis:")

            # Base composition
            base_counts = {base: sequence.count(base) for base in self.BASES}
            lines.append("  Base composition:")
            for base in self.BASES:
                count = base_counts[base]
                pct = (count / len(sequence)) * 100
                lines.append(f"    {base}: {count} ({pct:.1f}%)")

            # GC content
            gc_count = sequence.count('G') + sequence.count('C')
            gc_ratio = gc_count / len(sequence)
            lines.append(f"  GC content: {gc_ratio:.2%}")

        return '\n'.join(lines)


# Example usage and tests
if __name__ == '__main__':
    print("Testing DNA Sequence Generation Environment")
    print("=" * 60)

    # Create environment (paper specification)
    env = DNASequence(
        seq_length=15,
        objectives=['free_energy', 'num_base_pairs', 'inverse_length'],
        temperature=37.0,
        use_nupack=True
    )

    print(f"\nEnvironment properties:")
    print(f"  Sequence length: {env.seq_length}")
    print(f"  State dim: {env.state_dim}")
    print(f"  Num actions: {env.num_actions}")
    print(f"  Num objectives: {env.num_objectives}")
    print(f"  Objectives: {env.objectives}")

    # Test building a specific sequence
    print(f"\n{'='*60}")
    print("Building sequence: ACGTACGTTATA")
    print("=" * 60)

    state = env.get_initial_state()
    target_sequence = "ACGTACGTTATA"

    for base in target_sequence:
        action = env.BASE_TO_IDX[base]
        state, done = env.step(state, action)
        print(f"  Added '{base}', done={done}")

    print(f"\n{env.visualize_sequence(state)}")

    # Test random sampling
    print(f"\n{'='*60}")
    print("Random sequence sampling")
    print("=" * 60)

    state = env.sample_random_trajectory()
    sequence = env._state_to_sequence(state)

    print(f"\nRandom sequence: {sequence}")
    print(f"Length: {len(sequence)}")

    objectives = env.compute_objectives(state)
    print(f"\nObjectives:")
    for i, obj_name in enumerate(env.objectives):
        print(f"  {obj_name}: {objectives[i].item():.4f}")

    print("\n" + "=" * 60)
    print(" DNA Sequence environment test complete!")
    print("=" * 60)
