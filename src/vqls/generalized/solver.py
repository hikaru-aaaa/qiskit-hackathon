"""
Main DF-VQLS Solver

Orchestrates the entire DF-VQLS algorithm to solve linear systems.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import OptimizeResult
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .utils import (
    validate_matrix_size,
    calculate_qubits,
    calculate_num_parameters,
    scale_solution
)
from .ansatz import HardwareEfficientAnsatz
from .state_preparer import StatePreparer
from .circuit_builder import CircuitBuilder
from .cost_function import CostFunction
from .optimizer import Optimizer


class DFVQLSSolver:
    """
    Decomposition-Free Variational Quantum Linear Solver.
    
    Solves linear systems of the form Ku = f using variational quantum algorithms
    without requiring matrix decomposition.
    
    Supports 8×8 and 16×16 systems (and any 2^n size).
    """
    
    def __init__(
        self,
        matrix_size: int,
        num_layers: int = 3,
        optimizer_method: str = 'COBYLA',
        max_iter: int = 200,
        random_seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize DF-VQLS solver.
        
        Args:
            matrix_size: Size of the matrix (must be power of 2, e.g., 8, 16)
            num_layers: Number of ansatz layers
            optimizer_method: Optimization method ('COBYLA', 'BFGS', etc.)
            max_iter: Maximum number of optimization iterations
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress information
        
        Raises:
            ValueError: If matrix_size is not a power of 2
        """
        # Validate matrix size
        validate_matrix_size(matrix_size)
        
        self.matrix_size = matrix_size
        self.n_qubits = calculate_qubits(matrix_size)
        self.num_layers = num_layers
        self.verbose = verbose
        
        # Initialize components
        self.ansatz = HardwareEfficientAnsatz(
            num_qubits=self.n_qubits,
            num_layers=num_layers
        )
        
        self.state_preparer = StatePreparer()
        
        self.circuit_builder = CircuitBuilder(
            matrix_size=matrix_size,
            ansatz=self.ansatz
        )
        
        # Create simulator (statevector mode for exact simulation)
        self.simulator = AerSimulator(method='statevector')
        
        self.cost_function = CostFunction(
            circuit_builder=self.circuit_builder,
            state_preparer=self.state_preparer,
            ansatz=self.ansatz,
            simulator=self.simulator,
            verbose=verbose
        )
        
        self.optimizer = Optimizer(
            method=optimizer_method,
            max_iter=max_iter,
            random_seed=random_seed,
            verbose=verbose
        )
    
    def solve(
        self,
        K: np.ndarray,
        f: np.ndarray
    ) -> Tuple[np.ndarray, OptimizeResult]:
        """
        Solve the linear system Ku = f.
        
        Args:
            K: Coefficient matrix (N×N, must match matrix_size)
            f: Right-hand side vector (N×1)
        
        Returns:
            Tuple of (solution_vector, optimization_result)
        
        Raises:
            ValueError: If matrix/vector dimensions don't match
        """
        # Validate inputs
        if K.shape != (self.matrix_size, self.matrix_size):
            raise ValueError(
                f"Matrix K must be {self.matrix_size}×{self.matrix_size}, "
                f"got {K.shape}"
            )
        
        if f.shape not in [(self.matrix_size,), (self.matrix_size, 1)]:
            raise ValueError(
                f"Vector f must be {self.matrix_size}-dimensional, "
                f"got {f.shape}"
            )
        
        f = f.flatten()  # Ensure f is 1D
        
        if self.verbose:
            print("=" * 70)
            print(f"DF-VQLS for {self.matrix_size}×{self.matrix_size} System")
            print("=" * 70)
            print(f"Matrix size: {self.matrix_size}×{self.matrix_size}")
            print(f"Number of qubits: {self.n_qubits}")
            print(f"Ansatz layers: {self.num_layers}")
            print(f"Max iterations: {self.optimizer.max_iter}")
            print(f"Optimizer: {self.optimizer.method}")
            print("=" * 70 + "\n")
        
        # Initialize random parameters
        num_params = self.ansatz.num_parameters()
        initial_params = np.random.uniform(0, 2 * np.pi, num_params)
        
        # Create cost function wrapper
        def cost_fn(params, pbar=None):
            return self.cost_function.compute(params, K, f, pbar=pbar)
        
        # Run optimization
        result = self.optimizer.optimize(
            cost_function=cost_fn,
            initial_params=initial_params
        )
        
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Optimization complete: {result.nfev} iterations")
            print(f"Final cost: {result.fun:.6f}")
            print(f"Success: {result.success}")
            print("=" * 70)
        
        # Extract solution
        u_quantum = self._extract_solution(result.x)
        
        # Scale solution
        u_scaled = scale_solution(u_quantum, K, f)
        
        return u_scaled, result
    
    def _extract_solution(self, params: np.ndarray) -> np.ndarray:
        """
        Extract solution state |u(θ)⟩ from optimized parameters.
        
        Args:
            params: Optimized ansatz parameters
        
        Returns:
            Quantum state vector |u(θ)⟩
        """
        # Create circuit with optimized parameters
        circ = QuantumCircuit(self.n_qubits)
        qubits = list(range(self.n_qubits))
        circ = self.ansatz.apply(circ, qubits, params)
        circ.save_statevector()
        
        # Run simulation
        transpiled_circ = transpile(circ, self.simulator)
        result = self.simulator.run(transpiled_circ).result()
        u_theta = np.real(np.array(result.get_statevector(circ)))
        
        return u_theta

