"""
Molecule generation environment for multi-objective GFlowNets.

Based on the environment described in:
    Jain et al. "Multi-Objective GFlowNets" (ICML 2023)

The molecule generation task involves constructing molecules by adding
molecular fragments iteratively. Multiple objectives are defined as
molecular properties (QED, SA score, logP).
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.mogfn_pc import MultiObjectiveEnvironment

# Suppress RDKit warnings about invalid molecules
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
try:
    from rdkit import RDLogger
    # Disable RDKit warnings/errors about invalid molecules
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass


class MoleculeFragments(MultiObjectiveEnvironment):
    """
    Fragment-based molecule generation environment with multi-objective rewards.

    State: Partially constructed molecule represented as a list of fragment indices
    Actions: Add molecular fragment (0 to num_fragments-1) or DONE (terminate)

    The agent starts with an empty molecule and adds fragments one at a time
    until reaching the maximum number of fragments or choosing to terminate.

    Multiple objectives are defined by molecular properties:
    - QED: Quantitative Estimate of Drug-likeness
    - SA: Synthetic Accessibility score
    - logP: Lipophilicity (octanol-water partition coefficient)
    - MW: Molecular Weight
    """

    # Common molecular fragments (SMILES representations)
    # These are building blocks that can be combined to form molecules
    FRAGMENT_LIBRARY = {
        0: 'C',          # Methyl
        1: 'CC',         # Ethyl
        2: 'CCC',        # Propyl
        3: 'c1ccccc1',   # Benzene ring
        4: 'C(=O)',      # Carbonyl
        5: 'C(=O)O',     # Carboxyl
        6: 'N',          # Amine
        7: 'O',          # Hydroxyl
        8: 'C(=O)N',     # Amide
        9: 'c1ccc(N)cc1', # Aniline
        10: 'C(F)(F)F',  # Trifluoromethyl
        11: 'c1ccncc1',  # Pyridine
        12: 'C1CCCCC1',  # Cyclohexane
        13: 'c1ccc(O)cc1', # Phenol
        14: 'C#N',       # Nitrile
    }

    def __init__(self,
                max_fragments: int = 8,
                num_fragments_library: int = 15,
                objective_properties: Optional[List[str]] = None,
                use_rdkit: bool = True):
        """
        Args:
            max_fragments: Maximum number of fragments in a molecule
            num_fragments_library: Number of fragments in the library to use
            objective_properties: List of properties to use as objectives
                                Options: 'qed', 'sa', 'logp'
                                If None, uses ['qed', 'sa']
            use_rdkit: Whether to use RDKit for property calculation
                    If False, uses simple heuristics (for testing without RDKit)
        """
        self.max_fragments = max_fragments
        self.num_fragments_library = min(num_fragments_library, len(self.FRAGMENT_LIBRARY))
        self.use_rdkit = use_rdkit

        # Check if RDKit is available
        self._rdkit_available = False
        if use_rdkit:
            try:
                import rdkit
                from rdkit import Chem
                from rdkit.Chem import Descriptors, QED
                self._rdkit_available = True
            except ImportError:
                print("Warning: RDKit not available. Using simple heuristics for properties.")
                self.use_rdkit = False

        # Setup objective properties
        if objective_properties is None:
            objective_properties = ['qed', 'sa', 'logp'] if self._rdkit_available else ['length', 'diversity']

        self.objective_properties = objective_properties
        self._num_objectives = len(objective_properties)

        # State representation: [num_fragments, frag_0, frag_1, ..., frag_{max-1}]
        # where num_fragments is current count and frag_i is fragment index (-1 if empty)
        self._state_dim = 1 + max_fragments

        # Actions: 0 to num_fragments_library-1 (add fragment), num_fragments_library (DONE)
        self._num_actions = num_fragments_library + 1
        self.ACTION_DONE = num_fragments_library

    def get_initial_state(self) -> torch.Tensor:
        """
        Get initial state (empty molecule).

        Returns:
            state: [num_fragments=0, -1, -1, ..., -1] (all fragments empty)
        """
        state = torch.full((self._state_dim,), -1.0, dtype=torch.float32)
        state[0] = 0.0  # Number of fragments starts at 0
        return state

    def step(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, bool]:
        """
        Take action in environment (add fragment to molecule).

        Args:
            state: Current state [num_frags, frag_0, ..., frag_{max-1}]
            action: Fragment index to add (0 to num_fragments_library-1) or DONE

        Returns:
            next_state: Next state
            is_terminal: Whether episode is done
        """
        # Clone state to avoid modifying original
        next_state = state.clone()

        num_frags = int(state[0].item())

        if action == self.ACTION_DONE:
            # Terminal action
            is_terminal = True

        elif action < self.num_fragments_library:
            # Add fragment at current position
            if num_frags < self.max_fragments:
                next_state[1 + num_frags] = float(action)
                next_state[0] = float(num_frags + 1)  # Increment count
            is_terminal = (num_frags + 1) >= self.max_fragments

        else:
            raise ValueError(f"Invalid action: {action}")

        return next_state, is_terminal

    def get_valid_actions(self, state: torch.Tensor) -> List[int]:
        """
        Get valid actions for current state.

        Can always terminate. Can add fragments if not at maximum.
        """
        num_frags = int(state[0].item())

        valid_actions = [self.ACTION_DONE]  # Can always terminate

        if num_frags < self.max_fragments:
            # Can add any fragment
            valid_actions.extend(range(self.num_fragments_library))

        return valid_actions

    def _state_to_smiles(self, state: torch.Tensor) -> str:
        """
        Convert state tensor to SMILES string.

        Args:
            state: State tensor [num_frags, frag_0, ..., frag_{max-1}]

        Returns:
            smiles: SMILES string representation
        """
        num_frags = int(state[0].item())

        if num_frags == 0:
            return ""

        # Collect fragments
        fragments = []
        for i in range(num_frags):
            frag_idx = int(state[1 + i].item())
            if frag_idx >= 0 and frag_idx < self.num_fragments_library:
                fragments.append(self.FRAGMENT_LIBRARY[frag_idx])

        # Simple concatenation (in practice, would need proper chemical bonding)
        smiles = ''.join(fragments)

        # If RDKit is available, validate and sanitize the molecule
        if self._rdkit_available and smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is not None:
                    # Try to sanitize the molecule
                    # This will fix issues like invalid valences where possible
                    try:
                        Chem.SanitizeMol(mol)
                        # Return canonicalized SMILES
                        smiles = Chem.MolToSmiles(mol)
                    except:
                        # Sanitization failed - molecule is chemically invalid
                        # Return empty string to indicate invalid molecule
                        return ""
            except:
                # Parsing failed completely
                return ""

        return smiles

    def _compute_qed(self, smiles: str) -> float:
        """Compute QED (Quantitative Estimate of Drug-likeness)."""
        if not self._rdkit_available or not smiles:
            # Heuristic: longer molecules with diverse fragments tend to be more drug-like
            return min(1.0, len(smiles) / 50.0)

        try:
            from rdkit import Chem
            from rdkit.Chem import QED

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            return QED.qed(mol)
        except Exception:
            return 0.0

    def _compute_sa_score(self, smiles: str) -> float:
        """Compute Synthetic Accessibility score (lower is better, we return 1-normalized)."""
        if not self._rdkit_available or not smiles:
            # Heuristic: simpler molecules are more synthetically accessible
            return max(0.0, 1.0 - len(smiles) / 100.0)

        try:
            from rdkit import Chem
            # Note: SA score requires additional data files
            # For simplicity, we approximate based on complexity
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            # Simple approximation: based on number of rings and heteroatoms
            num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
            num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ['C', 'H'])

            # Lower complexity = higher score
            complexity = num_rings * 0.3 + num_hetero * 0.2
            sa_score = max(0.0, 1.0 - complexity / 5.0)
            return sa_score
        except Exception:
            return 0.0

    def _compute_logp(self, smiles: str) -> float:
        """Compute logP (lipophilicity), normalized to [0, 1]."""
        if not self._rdkit_available or not smiles:
            # Heuristic: more carbon atoms = higher logP
            return min(1.0, smiles.count('C') / 20.0)

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            logp = Descriptors.MolLogP(mol)
            # Normalize: typical range is -2 to 6, map to [0, 1]
            normalized_logp = (logp + 2) / 8.0
            return max(0.0, min(1.0, normalized_logp))
        except Exception:
            return 0.0

    def _compute_molecular_weight(self, smiles: str) -> float:
        """Compute molecular weight, normalized to [0, 1]."""
        if not self._rdkit_available or not smiles:
            # Heuristic: approximate by string length
            return min(1.0, len(smiles) / 100.0)

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            mw = Descriptors.MolWt(mol)
            # Normalize: typical drug MW is 200-500, map to [0, 1]
            normalized_mw = (mw - 100) / 600.0
            return max(0.0, min(1.0, normalized_mw))
        except Exception:
            return 0.0

    def _compute_fragment_diversity(self, state: torch.Tensor) -> float:
        """Compute diversity of fragments used (for testing without RDKit)."""
        num_frags = int(state[0].item())
        if num_frags == 0:
            return 0.0

        fragments = []
        for i in range(num_frags):
            frag_idx = int(state[1 + i].item())
            if frag_idx >= 0:
                fragments.append(frag_idx)

        # Diversity = fraction of unique fragments
        unique_frags = len(set(fragments))
        diversity = unique_frags / num_frags if num_frags > 0 else 0.0
        return diversity

    def compute_objectives(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute all objective values for terminal state.

        Args:
            state: Terminal state

        Returns:
            objectives: Tensor of shape (num_objectives,) with property values
        """
        smiles = self._state_to_smiles(state)

        objectives = []
        for prop in self.objective_properties:
            if prop == 'qed':
                objectives.append(self._compute_qed(smiles))
            elif prop == 'sa':
                objectives.append(self._compute_sa_score(smiles))
            elif prop == 'logp':
                objectives.append(self._compute_logp(smiles))
            elif prop == 'mw':
                objectives.append(self._compute_molecular_weight(smiles))
            elif prop == 'length':
                # Simple fallback: molecule length
                num_frags = int(state[0].item())
                objectives.append(num_frags / self.max_fragments)
            elif prop == 'diversity':
                objectives.append(self._compute_fragment_diversity(state))
            else:
                objectives.append(0.0)

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

    def get_fragment_name(self, idx: int) -> str:
        """Get human-readable name for fragment."""
        fragment_names = {
            0: 'Methyl',
            1: 'Ethyl',
            2: 'Propyl',
            3: 'Benzene',
            4: 'Carbonyl',
            5: 'Carboxyl',
            6: 'Amine',
            7: 'Hydroxyl',
            8: 'Amide',
            9: 'Aniline',
            10: 'Trifluoromethyl',
            11: 'Pyridine',
            12: 'Cyclohexane',
            13: 'Phenol',
            14: 'Nitrile',
        }
        return fragment_names.get(idx, f'Fragment_{idx}')

    def visualize_molecule(self, state: torch.Tensor, save_path: Optional[str] = None):
        """
        Visualize molecule structure.

        Args:
            state: State tensor
            save_path: Path to save figure
        """
        smiles = self._state_to_smiles(state)

        if not self._rdkit_available:
            print(f"SMILES: {smiles}")
            print("RDKit not available for visualization")
            return

        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                return

            # Generate 2D coordinates
            from rdkit.Chem import AllChem
            AllChem.Compute2DCoords(mol)

            # Draw molecule
            img = Draw.MolToImage(mol, size=(400, 400))

            if save_path:
                img.save(save_path)
                print(f"Saved molecule image to {save_path}")
            else:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f'SMILES: {smiles}')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error visualizing molecule: {e}")
            print(f"SMILES: {smiles}")

    def visualize_samples(self,
                        objectives: torch.Tensor,
                        states: Optional[List[torch.Tensor]] = None,
                        save_path: Optional[str] = None):
        """
        Visualize sampled molecules in objective space.

        Args:
            objectives: Tensor of shape (N, num_objectives)
            states: Optional list of state tensors
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt

        objectives_np = objectives.detach().cpu().numpy()

        if self._num_objectives == 2:
            # 2D scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(objectives_np[:, 0], objectives_np[:, 1],
                    alpha=0.6, s=50, c='blue')

            # Annotate some points with SMILES
            if states and len(states) <= 10:
                for i in range(min(len(states), 10)):
                    smiles = self._state_to_smiles(states[i])
                    if smiles:
                        plt.annotate(smiles[:15],
                                (objectives_np[i, 0], objectives_np[i, 1]),
                                fontsize=7, alpha=0.7,
                                xytext=(5, 5), textcoords='offset points')

            plt.xlabel(f'Objective 1: {self.objective_properties[0].upper()}')
            plt.ylabel(f'Objective 2: {self.objective_properties[1].upper()}')
            plt.title(f'Molecule Solutions (max_frags={self.max_fragments})')
            plt.grid(True, alpha=0.3)

        elif self._num_objectives >= 3:
            # 3D scatter plot
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives_np[:, 0], objectives_np[:, 1], objectives_np[:, 2],
                    alpha=0.6, s=50, c='blue')
            ax.set_xlabel(self.objective_properties[0].upper())
            ax.set_ylabel(self.objective_properties[1].upper())
            ax.set_zlabel(self.objective_properties[2].upper())
            ax.set_title('Molecule Solutions (3D)')

        else:
            # Single objective - histogram
            plt.figure(figsize=(8, 6))
            plt.hist(objectives_np[:, 0], bins=30, alpha=0.7, color='blue')
            plt.xlabel(f'{self.objective_properties[0].upper()}')
            plt.ylabel('Count')
            plt.title('Molecule Solutions Distribution')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()


def test_molecules():
    """Test MoleculeFragments environment."""

    print("Testing Molecule Generation Environment")
    print("=" * 60)

    # Create environment (works without RDKit using heuristics)
    env = MoleculeFragments(
        max_fragments=6,
        num_fragments_library=10,
        objective_properties=['length', 'diversity']
    )

    print(f"Max fragments: {env.max_fragments}")
    print(f"Fragment library size: {env.num_fragments_library}")
    print(f"State dim: {env.state_dim}")
    print(f"Num actions: {env.num_actions}")
    print(f"Num objectives: {env.num_objectives}")
    print(f"Objective properties: {env.objective_properties}")
    print(f"RDKit available: {env._rdkit_available}")

    # Test initial state
    state = env.get_initial_state()
    print(f"\nInitial state: {state}")
    print(f"SMILES: '{env._state_to_smiles(state)}'")

    # Build a test molecule
    print("\nBuilding molecule with fragments: Benzene-Amine-Carbonyl")
    actions = [3, 6, 4]  # Benzene, Amine, Carbonyl

    for i, action in enumerate(actions):
        valid_actions = env.get_valid_actions(state)
        print(f"  Step {i}: Add {env.get_fragment_name(action)}, valid_actions={len(valid_actions)}")

        state, done = env.step(state, action)
        smiles = env._state_to_smiles(state)
        print(f"    SMILES: '{smiles}', done={done}")

    # Compute objectives for final state
    print(f"\nFinal molecule:")
    print(f"  SMILES: '{env._state_to_smiles(state)}'")

    objectives = env.compute_objectives(state)
    print(f"  Objectives: {objectives}")

    for i, prop in enumerate(env.objective_properties):
        print(f"    {prop}: {objectives[i].item():.3f}")

    # Test different molecule constructions
    print("\n" + "="*60)
    print("Testing different molecules:")
    print("="*60)

    test_sequences = [
        [0, 0, 0, 0],           # Four methyls
        [3, 3],                  # Two benzene rings
        [3, 6, 7],              # Benzene-Amine-Hydroxyl
        [1, 4, 5],              # Ethyl-Carbonyl-Carboxyl
    ]

    for seq in test_sequences:
        state = env.get_initial_state()
        for action in seq:
            state, _ = env.step(state, action)

        smiles = env._state_to_smiles(state)
        objectives = env.compute_objectives(state)

        print(f"\nFragments: {[env.get_fragment_name(a) for a in seq]}")
        print(f"  SMILES: '{smiles}'")
        print(f"  Objectives: {objectives.numpy()}")

    # If RDKit is available, test with molecular properties
    if env._rdkit_available:
        print("\n" + "="*60)
        print("Testing with RDKit properties:")
        print("="*60)

        env_rdkit = MoleculeFragments(
            max_fragments=6,
            num_fragments_library=10,
            objective_properties=['qed', 'sa', 'logp']
        )

        state = env_rdkit.get_initial_state()
        actions = [3, 6, 7]  # Benzene-Amine-Hydroxyl

        for action in actions:
            state, _ = env_rdkit.step(state, action)

        smiles = env_rdkit._state_to_smiles(state)
        objectives = env_rdkit.compute_objectives(state)

        print(f"Molecule: {smiles}")
        print(f"Properties:")
        for i, prop in enumerate(env_rdkit.objective_properties):
            print(f"  {prop.upper()}: {objectives[i].item():.3f}")

    print("\n" + "="*60)
    print("Molecule environment test completed!")
    print("="*60)


if __name__ == '__main__':
    test_molecules()
