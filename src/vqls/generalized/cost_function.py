"""
Cost function computation for DF-VQLS

Computes the global cost function CG(θ) using swap test circuits.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from typing import Optional

from .circuit_builder import CircuitBuilder
from .state_preparer import StatePreparer
from .ansatz import Ansatz


def extract_P0_from_statevector(statevector: np.ndarray) -> float:
    """
    Extract probability of measuring ancilla in |0⟩ state.
    
    For a statevector, P(0) is the sum of probabilities of all even-indexed
    basis states (where ancilla qubit is in |0⟩ state).
    
    Args:
        statevector: Quantum statevector from simulation
    
    Returns:
        Probability P(0) of measuring ancilla in |0⟩ state
    """
    P0 = 0.0
    for i in range(len(statevector)):
        if i % 2 == 0:  # Even indices correspond to ancilla = |0⟩
            P0 += abs(statevector[i]) ** 2
    return P0


class CostFunction:
    """
    Computes the global cost function CG(θ) for DF-VQLS.
    
    Cost function: CG(θ) = 1 - |⟨f|K|u(θ)⟩|² / ⟨u(θ)|K^T K|u(θ)⟩
    
    When CG(θ) = 0, we have found the exact solution.
    """
    
    def __init__(
        self,
        circuit_builder: CircuitBuilder,
        state_preparer: StatePreparer,
        ansatz: Ansatz,
        simulator: AerSimulator,
        verbose: bool = False
    ):
        """
        Initialize cost function.
        
        Args:
            circuit_builder: Circuit builder instance
            state_preparer: State preparer instance
            ansatz: Ansatz instance
            simulator: Qiskit Aer simulator
            verbose: Whether to print detailed cost information
        """
        self.circuit_builder = circuit_builder
        self.state_preparer = state_preparer
        self.ansatz = ansatz
        self.simulator = simulator
        self.verbose = verbose
        
        self.n_qubits = circuit_builder.n_qubits
    
    def _compute_u_theta(self, params: np.ndarray) -> np.ndarray:
        """
        Compute |u(θ)⟩ by running the ansatz.
        
        Args:
            params: Ansatz parameters
        
        Returns:
            Quantum state vector |u(θ)⟩
        """
        # Create temporary circuit for ansatz
        temp_circ = QuantumCircuit(self.n_qubits)
        qubits = list(range(self.n_qubits))
        temp_circ = self.ansatz.apply(temp_circ, qubits, params)
        temp_circ.save_statevector()
        
        # Run simulation
        transpiled_circ = transpile(temp_circ, self.simulator)
        result = self.simulator.run(transpiled_circ).result()
        u_theta = np.asarray(result.get_statevector(temp_circ))
        
        return u_theta
    
    def compute(
        self,
        params: np.ndarray,
        K: np.ndarray,
        f: np.ndarray,
        pbar: Optional[object] = None
    ) -> float:
        """
        Compute the global cost function CG(θ).
        
        Args:
            params: Ansatz parameters
            K: Coefficient matrix
            f: Right-hand side vector
            pbar: Optional tqdm progress bar for updating
        
        Returns:
            Cost function value (0 means perfect solution)
        """
        # Prepare states
        vec_K, norm_K = self.state_preparer.prepare_matrix(K)
        vec_KT, _ = self.state_preparer.prepare_matrix_transpose(K)
        f_norm = self.state_preparer.prepare_vector(f)
        
        # Compute |u(θ)⟩
        u_theta = self._compute_u_theta(params)
        
        # ===== Compute Numerator: |⟨f|K|u(θ)⟩|² =====
        circ_num = self.circuit_builder.build_numerator_circuit(
            vec_K, f_norm, u_theta
        )
        circ_num.save_statevector()
        
        # Run numerator circuit
        transpiled_num = transpile(circ_num, self.simulator)
        result_num = self.simulator.run(transpiled_num).result()
        sv_num = np.asarray(result_num.get_statevector(circ_num))
        
        # Extract P(0) from statevector
        P0_num = extract_P0_from_statevector(sv_num)
        
        # Calculate inner product squared: |⟨ψ₁|ψ₂⟩|² = 2P(0) - 1
        inner_product_squared_num = abs(2 * P0_num - 1)
        
        # Numerator needs |⟨vec(K)|u,f⟩|² (squared overlap)
        numerator = (norm_K ** 2) * inner_product_squared_num
        
        # ===== Compute Denominator: ⟨u(θ)|K^T K|u(θ)⟩ =====
        circ_den = self.circuit_builder.build_denominator_circuit(
            vec_K, vec_KT, u_theta
        )
        circ_den.save_statevector()
        
        # Run denominator circuit
        transpiled_den = transpile(circ_den, self.simulator)
        result_den = self.simulator.run(transpiled_den).result()
        sv_den = np.asarray(result_den.get_statevector(circ_den))
        
        # Extract P(0) from statevector
        P0_den = extract_P0_from_statevector(sv_den)
        
        # Calculate inner product squared
        inner_product_squared_den = abs(2 * P0_den - 1)
        
        # Denominator needs ⟨u|K^T K|u⟩ = ⟨u⊗vec(K^T)|vec(K)⊗u⟩ (inner product, NOT squared!)
        # Swap test gives |⟨ψ₁|ψ₂⟩|², so take sqrt to get the inner product value
        inner_product_den = np.sqrt(inner_product_squared_den)
        denominator = (norm_K ** 2) * inner_product_den
        
        # Compute cost function
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        cost = 1.0 - (numerator / (denominator + epsilon))
        
        # Update progress bar if provided
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({'cost': f'{cost:.6f}'})
        
        # Print detailed information if verbose
        if self.verbose and pbar is None:
            print(
                f"Cost: {cost:.6f} | Num: {numerator:.6f} | "
                f"Den: {denominator:.6f} | P0_num: {P0_num:.4f} | "
                f"P0_den: {P0_den:.4f}"
            )
        
        return cost

