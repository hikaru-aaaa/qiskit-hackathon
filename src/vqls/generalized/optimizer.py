"""
Optimizer for DF-VQLS

Handles optimization of variational parameters using classical optimizers.
Supports both gradient-based (BFGS, L-BFGS-B, CG) and gradient-free (COBYLA) methods.
"""

import numpy as np
import random
from typing import Optional, Callable, Union, Tuple
from scipy.optimize import minimize, OptimizeResult
from tqdm import tqdm


# Define optimizer categories
GRADIENT_METHODS = ['BFGS', 'L-BFGS-B', 'CG']
GRADIENT_FREE_METHODS = ['COBYLA', 'NELDER-MEAD', 'SPSA']


class Optimizer:
    """
    Optimizer for variational quantum algorithms.

    Supports both gradient-based (BFGS, L-BFGS-B, CG) and gradient-free (COBYLA) methods
    with progress bar integration.
    """

    def __init__(
        self,
        method: str = 'COBYLA',
        max_iter: int = 200,
        random_seed: Optional[int] = None,
        verbose: bool = True,
        gtol: float = 1e-3,
        use_gradient: Optional[bool] = None
    ):
        """
        Initialize optimizer.

        Args:
            method: Optimization method ('COBYLA', 'BFGS', 'L-BFGS-B', 'CG', etc.)
            max_iter: Maximum number of iterations
            random_seed: Random seed for reproducibility
            verbose: Whether to show progress bar
            gtol: Gradient tolerance for gradient-based methods (default: 1e-3, from paper)
            use_gradient: Whether to use gradient information. If None, auto-detected based on method.
        """
        self.method = method
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.verbose = verbose
        self.gtol = gtol

        # Auto-detect gradient usage if not specified
        if use_gradient is None:
            self.use_gradient = method in GRADIENT_METHODS
        else:
            self.use_gradient = use_gradient

        # Validation
        if self.use_gradient and method not in GRADIENT_METHODS:
            raise ValueError(
                f"Method {method} does not support gradients. "
                f"Choose from: {GRADIENT_METHODS}"
            )

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def optimize(
        self,
        cost_function,  # CostFunction object with compute() and compute_with_gradient() methods
        initial_params: np.ndarray,
        K: np.ndarray,
        f: np.ndarray,
        pbar: Optional[tqdm] = None,
        track_iterations: bool = False
    ) -> OptimizeResult:
        """
        Optimize variational parameters.

        Args:
            cost_function: CostFunction object with compute() and compute_with_gradient() methods
            initial_params: Initial parameter values
            K: Coefficient matrix (needed for gradient computation)
            f: Right-hand side vector (needed for gradient computation)
            pbar: Optional progress bar (if None, will create one if verbose=True)
            track_iterations: If True, store iteration history (params and cost at each iteration)

        Returns:
            Optimization result from scipy.optimize.minimize
            If track_iterations=True, result will have 'iteration_history' attribute
        """
        # Create progress bar if needed
        if pbar is None and self.verbose:
            pbar = tqdm(
                total=self.max_iter,
                desc=f"Optimizing ({self.method})",
                unit="iter",
                disable=not self.verbose
            )

        # Initialize iteration history if tracking is enabled
        iteration_history = [] if track_iterations else None

        if self.use_gradient:
            # GRADIENT-BASED OPTIMIZATION
            def wrapped_cost_and_grad(params):
                """Wrapper that returns both cost and gradient."""
                cost, grad = cost_function.compute_with_gradient(
                    params, K, f,
                    pbar=None,  # Don't pass pbar to inner computation
                    use_parallel=False
                )

                # Store iteration data if tracking is enabled
                if iteration_history is not None:
                    iteration_history.append({
                        'iteration': len(iteration_history),
                        'params': params.copy(),
                        'cost': float(cost),
                        'grad_norm': float(np.linalg.norm(grad))
                    })

                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        'cost': f'{cost:.6f}',
                        'grad_norm': f'{np.linalg.norm(grad):.6f}'
                    })

                return cost, grad

            # Run optimization with gradient
            result = minimize(
                fun=wrapped_cost_and_grad,
                x0=initial_params,
                method=self.method,
                jac=True,  # IMPORTANT: Tell minimize that fun returns (cost, grad)
                options={
                    'maxiter': self.max_iter,
                    'gtol': self.gtol,
                    'disp': self.verbose
                }
            )
        else:
            # GRADIENT-FREE OPTIMIZATION
            def wrapped_cost(params):
                cost = cost_function.compute(params, K, f, pbar=None)

                # Store iteration data if tracking is enabled
                if iteration_history is not None:
                    iteration_history.append({
                        'iteration': len(iteration_history),
                        'params': params.copy(),
                        'cost': float(cost)
                    })

                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({'cost': f'{cost:.6f}'})

                return cost

            # Run optimization without gradient
            result = minimize(
                fun=wrapped_cost,
                x0=initial_params,
                method=self.method,
                options={'maxiter': self.max_iter}
            )

        # Close progress bar if we created it
        if pbar is not None and self.verbose:
            pbar.close()

        # Attach iteration history to result if tracking was enabled
        if track_iterations:
            result.iteration_history = iteration_history

        return result

