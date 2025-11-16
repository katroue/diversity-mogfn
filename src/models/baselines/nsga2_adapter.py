"""
NSGA-II Baseline for Multi-Objective Optimization.

Adapts the pymoo NSGA-II algorithm to work with MultiObjectiveEnvironment.
NSGA-II is a classic multi-objective genetic algorithm that uses:
- Non-dominated sorting
- Crowding distance for diversity
- Tournament selection

Installation:
    pip install pymoo

Reference:
    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
    A fast and elitist multiobjective genetic algorithm: NSGA-II.
    IEEE transactions on evolutionary computation, 6(2), 182-197.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logging.warning(
        "pymoo not installed. Install with: pip install pymoo\n"
        "NSGA2Adapter will not be functional until pymoo is installed."
    )

logger = logging.getLogger(__name__)


if PYMOO_AVAILABLE:
    class EnvironmentProblem(Problem):
        """
        Adapts MultiObjectiveEnvironment to pymoo Problem interface.

        Encodes trajectories as continuous decision variables and evaluates
        them by simulating the trajectory through the environment.
        """

        def __init__(
            self,
            env,
            max_steps: int = 100,
            seed: Optional[int] = None
        ):
            """
            Args:
                env: MultiObjectiveEnvironment instance
                max_steps: Maximum trajectory length
                seed: Random seed for reproducibility
            """
            self.env = env
            self.max_steps = max_steps
            self.rng = np.random.RandomState(seed)

            # Determine decision variable bounds
            # We encode trajectory as sequence of action probabilities
            n_var = max_steps  # One decision variable per timestep
            xl = np.zeros(n_var)  # Lower bounds
            xu = np.ones(n_var)   # Upper bounds

            # Determine number of objectives
            # Sample a trajectory to get objective dimensionality
            sample_state = env.get_initial_state()
            import torch
            objectives_tensor = env.compute_objectives(sample_state)
            sample_objectives = objectives_tensor.detach().cpu().numpy() if torch.is_tensor(objectives_tensor) else np.array(objectives_tensor)
            n_obj = len(sample_objectives)

            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                xl=xl,
                xu=xu
            )

            logger.info(
                f"Created EnvironmentProblem: {n_var} vars, {n_obj} objectives, "
                f"max_steps={max_steps}"
            )

        def _evaluate(self, X, out, *args, **kwargs):
            """
            Evaluate population of solutions.

            Args:
                X: Decision variables of shape (pop_size, n_var)
                   Each row is a sequence of [0,1] values used to select actions
                out: Dictionary to store objectives
            """
            objectives = []

            for x in X:
                obj = self._evaluate_trajectory(x)
                objectives.append(obj)

            # NSGA-II minimizes by default, but we want to maximize
            # So we negate the objectives
            out["F"] = -np.array(objectives)

        def _evaluate_trajectory(self, x: np.ndarray) -> np.ndarray:
            """
            Evaluate a single trajectory encoded as decision variables.

            Args:
                x: Decision variables of shape (n_var,)
                   Values in [0,1] used to select actions

            Returns:
                Objectives array of shape (n_obj,)
            """
            state = self.env.get_initial_state()

            for step in range(self.max_steps):
                # Get valid actions
                valid_actions = self.env.get_valid_actions(state)
                if len(valid_actions) == 0:
                    break

                # Use decision variable to select action
                # x[step] ∈ [0,1] maps to index in valid_actions
                action_idx = int(x[step] * len(valid_actions))
                action_idx = min(action_idx, len(valid_actions) - 1)  # Clamp
                action = valid_actions[action_idx]

                # Take action
                next_state, done = self.env.step(state, action)
                state = next_state

                if done:
                    break

            # Get final objectives (convert to numpy)
            import torch
            objectives_tensor = self.env.compute_objectives(state)
            objectives = objectives_tensor.detach().cpu().numpy() if torch.is_tensor(objectives_tensor) else np.array(objectives_tensor)
            return objectives


class NSGA2Adapter:
    """
    NSGA-II baseline for multi-objective optimization.

    Uses the pymoo library implementation of NSGA-II to optimize
    trajectories in a MultiObjectiveEnvironment.

    Args:
        env: MultiObjectiveEnvironment instance
        pop_size: Population size (default: 100)
        max_steps: Maximum trajectory length (default: 100)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        env,
        pop_size: int = 100,
        max_steps: int = 100,
        seed: Optional[int] = None
    ):
        if not PYMOO_AVAILABLE:
            raise ImportError(
                "pymoo is required for NSGA2Adapter. "
                "Install with: pip install pymoo"
            )

        self.env = env
        self.pop_size = pop_size
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Create problem
        self.problem = EnvironmentProblem(
            env=env,
            max_steps=max_steps,
            seed=seed
        )

        # Create algorithm
        self.algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        # Results storage
        self.result = None
        self.objectives_history: List[np.ndarray] = []
        self.history: Dict[str, List] = {
            'generation': [],
            'n_evals': [],
            'pareto_size': [],
            'hypervolume': []
        }

        logger.info(
            f"Initialized NSGA2Adapter with pop_size={pop_size}, "
            f"max_steps={max_steps}, seed={seed}"
        )

    def train(
        self,
        num_iterations: int,
        batch_size: int = None,  # Ignored for NSGA-II
        log_interval: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train NSGA-II for specified number of generations.

        Args:
            num_iterations: Number of generations to run
            batch_size: Ignored (NSGA-II uses pop_size)
            log_interval: Log progress every N generations
            **kwargs: Additional arguments passed to pymoo minimize()

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting NSGA-II optimization: {num_iterations} generations")
        logger.info("Note: There will be a brief silent period during initialization...")

        # Set up termination criterion
        termination = get_termination("n_gen", num_iterations)

        # Run optimization
        import sys
        logger.info("Running pymoo minimize() - progress updates will appear shortly...")
        sys.stdout.flush()  # Force output to appear immediately

        self.result = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=self.seed,
            verbose=False,
            save_history=True,
            **kwargs
        )

        # Extract history
        for i, algo in enumerate(self.result.history):
            if i % log_interval == 0 or i == len(self.result.history) - 1:
                # Get Pareto front (negate back to maximize)
                pareto_front = -algo.opt.get("F")

                self.history['generation'].append(i)
                self.history['n_evals'].append(algo.evaluator.n_eval)
                self.history['pareto_size'].append(len(pareto_front))

                # Compute hypervolume if available
                try:
                    from pymoo.indicators.hv import HV
                    # Use nadir point as reference
                    ref_point = np.max(pareto_front, axis=0) + 1.0
                    hv = HV(ref_point=ref_point)
                    hypervolume = hv(pareto_front)
                    self.history['hypervolume'].append(hypervolume)
                except Exception as e:
                    logger.debug(f"Could not compute hypervolume: {e}")
                    self.history['hypervolume'].append(0.0)

                logger.info(
                    f"Generation {i}/{num_iterations}: "
                    f"Evals {algo.evaluator.n_eval}, "
                    f"Pareto size {len(pareto_front)}, "
                    f"HV {self.history['hypervolume'][-1]:.4f}"
                )

        # Store all evaluated objectives (negate back to maximize)
        all_F = -self.result.history[-1].pop.get("F")
        self.objectives_history = [obj for obj in all_F]

        logger.info(
            f"NSGA-II complete: {len(self.result.opt)} Pareto optimal solutions, "
            f"{self.result.algorithm.n_gen} generations, "
            f"{self.result.algorithm.evaluator.n_eval} evaluations"
        )

        return self.history

    def sample(
        self,
        num_samples: int,
        preference: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample solutions from the final population.

        Args:
            num_samples: Number of solutions to sample
            preference: Optional preference vector (not used by NSGA-II)

        Returns:
            Tuple of (objectives, decision_vars)
            - objectives: (num_samples, num_objectives)
            - decision_vars: (num_samples, n_var)
        """
        if self.result is None:
            raise ValueError("Must call train() before sampling")

        if preference is not None:
            logger.warning("NSGA2Adapter ignores preference parameter")

        # Get all solutions from final population
        all_objectives = -self.result.pop.get("F")  # Negate back to maximize
        all_X = self.result.pop.get("X")

        # Sample randomly
        n_available = len(all_objectives)
        if num_samples > n_available:
            logger.warning(
                f"Requested {num_samples} samples but only {n_available} available. "
                f"Returning all available solutions."
            )
            num_samples = n_available

        indices = self.rng.choice(n_available, size=num_samples, replace=False)

        return all_objectives[indices], all_X[indices]

    def get_pareto_front(self) -> np.ndarray:
        """
        Get the Pareto front from NSGA-II results.

        Returns:
            Pareto front objectives of shape (num_pareto, num_objectives)
        """
        if self.result is None:
            raise ValueError("Must call train() before getting Pareto front")

        # Negate back to maximize
        pareto_front = -self.result.opt.get("F")
        return pareto_front

    def get_all_objectives(self) -> np.ndarray:
        """
        Get all evaluated objectives from final population.

        Returns:
            Array of shape (pop_size, num_objectives)
        """
        if self.result is None:
            raise ValueError("Must call train() before getting objectives")

        # Negate back to maximize
        all_objectives = -self.result.pop.get("F")
        return all_objectives

    def reset(self):
        """Reset the algorithm (requires re-initialization)."""
        self.result = None
        self.objectives_history.clear()
        self.history = {
            'generation': [],
            'n_evals': [],
            'pareto_size': [],
            'hypervolume': []
        }
        logger.info("NSGA2Adapter reset")

    def __repr__(self) -> str:
        return (
            f"NSGA2Adapter(env={self.env.__class__.__name__}, "
            f"pop_size={self.pop_size}, "
            f"max_steps={self.max_steps})"
        )