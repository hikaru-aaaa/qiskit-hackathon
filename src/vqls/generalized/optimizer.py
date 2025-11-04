"""
Optimizer for DF-VQLS

Handles optimization of variational parameters using classical optimizers.
"""

import numpy as np
import random
from typing import Optional, Callable
from scipy.optimize import minimize, OptimizeResult
from tqdm import tqdm


class Optimizer:
    """
    Optimizer for variational quantum algorithms.
    
    Supports multiple optimization methods (COBYLA, BFGS, etc.)
    with progress bar integration.
    """
    
    def __init__(
        self,
        method: str = 'COBYLA',
        max_iter: int = 200,
        random_seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize optimizer.
        
        Args:
            method: Optimization method ('COBYLA', 'BFGS', etc.)
            max_iter: Maximum number of iterations
            random_seed: Random seed for reproducibility
            verbose: Whether to show progress bar
        """
        self.method = method
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.verbose = verbose
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def optimize(
        self,
        cost_function: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        pbar: Optional[tqdm] = None
    ) -> OptimizeResult:
        """
        Optimize variational parameters.
        
        Args:
            cost_function: Cost function to minimize
            initial_params: Initial parameter values
            pbar: Optional progress bar (if None, will create one if verbose=True)
        
        Returns:
            Optimization result from scipy.optimize.minimize
        """
        # Create progress bar if needed
        if pbar is None and self.verbose:
            pbar = tqdm(
                total=self.max_iter,
                desc="Optimizing",
                unit="iter",
                disable=not self.verbose
            )
        
        # Wrap cost function to update progress bar
        def wrapped_cost(params):
            return cost_function(params, pbar=pbar)
        
        # Run optimization
        result = minimize(
            fun=wrapped_cost,
            x0=initial_params,
            method=self.method,
            options={'maxiter': self.max_iter}
        )
        
        # Close progress bar if we created it
        if pbar is not None and self.verbose:
            pbar.close()
        
        return result

