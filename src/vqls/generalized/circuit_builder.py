"""
Quantum circuit builder for DF-VQLS

Constructs numerator and denominator circuits for cost function computation.
"""

from typing import List
import numpy as np
from qiskit import QuantumCircuit

from .ansatz import Ansatz
from .utils import calculate_qubits


def apply_swap_test(
    circ: QuantumCircuit,
    ancilla: int,
    reg1_qubits: List[int],
    reg2_qubits: List[int]
) -> QuantumCircuit:
    """
    Apply swap test circuit to compute |⟨ψ₁|ψ₂⟩|².
    
    The swap test measures the overlap between two quantum states using an
    ancilla qubit. The probability of measuring |0⟩ on the ancilla gives:
    P(0) = (1 + |⟨ψ₁|ψ₂⟩|²) / 2
    Therefore: |⟨ψ₁|ψ₂⟩|² = 2P(0) - 1
    
    Args:
        circ: Quantum circuit
        ancilla: Index of ancilla qubit
        reg1_qubits: List of qubit indices for first register
        reg2_qubits: List of qubit indices for second register
    
    Returns:
        Modified circuit with swap test applied
    """
    assert len(reg1_qubits) == len(reg2_qubits), \
        "Register sizes must match for swap test"
    
    # Hadamard on ancilla to create superposition
    circ.h(ancilla)
    
    # Controlled-SWAP between corresponding qubits in the two registers
    for q1, q2 in zip(reg1_qubits, reg2_qubits):
        circ.cswap(ancilla, q1, q2)
    
    # Final Hadamard on ancilla
    circ.h(ancilla)
    
    return circ


class CircuitBuilder:
    """
    Builds quantum circuits for DF-VQLS cost function computation.
    
    Constructs:
    - Numerator circuit: |⟨f|K|u(θ)⟩|²
    - Denominator circuit: ⟨u(θ)|K^T K|u(θ)⟩
    """
    
    def __init__(self, matrix_size: int, ansatz: Ansatz):
        """
        Initialize circuit builder.
        
        Args:
            matrix_size: Size of the matrix (must be power of 2)
            ansatz: Ansatz instance for preparing |u(θ)⟩
        """
        self.matrix_size = matrix_size
        self.n_qubits = calculate_qubits(matrix_size)  # log2(N)
        self.ansatz = ansatz
        
        # Calculate total qubits needed for circuits
        # Numerator: 1 ancilla + 2*n_qubits (vec(K)) + 2*n_qubits (u⊗f) = 4*n_qubits + 1
        # Denominator: 1 ancilla + (n_qubits + 2*n_qubits) (u⊗vec(K^T)) + (2*n_qubits + n_qubits) (vec(K)⊗u) = 1 + 6*n_qubits
        self.num_qubits_numerator = 4 * self.n_qubits + 1
        self.num_qubits_denominator = 1 + 6 * self.n_qubits
    
    def build_numerator_circuit(
        self,
        vec_K: np.ndarray,
        f_norm: np.ndarray,
        u_theta: np.ndarray
    ) -> QuantumCircuit:
        """
        Build circuit for computing CG numerator: |⟨f|K|u(θ)⟩|²
        
        Uses vectorization: |⟨f|K|u(θ)⟩|² = ||K||² |⟨vec(K)|u(θ),f⟩|²
        
        According to the paper's circuit (a):
        - |ψ₁⟩ = |vec(K)⟩_nq ⊗ |0⟩_nq
        - |ψ₂⟩ = |u(θ)⟩_nq ⊗ |f⟩_nq
        - Swap test between two n_q-qubit register pairs
        
        Qubit layout:
        - q[0]: ancilla for swap test
        - q[1 : n_qubits+1]: first n_q qubits of |vec(K)⟩
        - q[n_qubits+1 : 2*n_qubits+1]: remaining n_q qubits of |vec(K)⟩ (or |0⟩)
        - q[2*n_qubits+1 : 3*n_qubits+1]: |u(θ)⟩
        - q[3*n_qubits+1 : 4*n_qubits+1]: |f⟩
        
        Args:
            vec_K: Normalized vectorized matrix |vec(K)⟩ (2^n_q dimensional)
            f_norm: Normalized right-hand side vector |f⟩
            u_theta: Pre-computed |u(θ)⟩ state vector
        
        Returns:
            Quantum circuit for CG numerator (for statevector simulation)
        """
        num_qubits = self.num_qubits_numerator
        circ = QuantumCircuit(num_qubits)
        
        # Calculate qubit ranges according to paper's circuit (a)
        # |ψ₁⟩ = |vec(K)⟩_nq ⊗ |0⟩_nq
        vec_K_first_start = 1
        vec_K_first_end = 1 + self.n_qubits
        vec_K_second_start = vec_K_first_end
        vec_K_second_end = 1 + 2 * self.n_qubits
        
        # |ψ₂⟩ = |u(θ)⟩_nq ⊗ |f⟩_nq
        u_theta_start = vec_K_second_end
        u_theta_end = u_theta_start + self.n_qubits
        f_start = u_theta_end
        f_end = num_qubits
        
        # Prepare |vec(K)⟩ on first n_q qubits [1, n_qubits]
        # Note: vec_K is 2^n_q dimensional, so we prepare it on 2*n_qubits
        # But according to paper, we split it into two n_q-qubit registers
        vec_K_first_qubits = list(range(vec_K_first_start, vec_K_first_end))
        vec_K_second_qubits = list(range(vec_K_second_start, vec_K_second_end))
        
        # Prepare |vec(K)⟩ on 2*n_qubits (first n_q qubits + remaining n_q qubits)
        vec_K_all_qubits = list(range(vec_K_first_start, vec_K_second_end))
        circ.initialize(vec_K, vec_K_all_qubits)
        
        # Prepare |u(θ)⟩ on qubits [2*n_qubits+1, 3*n_qubits]
        u_theta_qubits = list(range(u_theta_start, u_theta_end))
        circ.initialize(u_theta, u_theta_qubits)
        
        # Prepare |f⟩ on qubits [3*n_qubits+1, 4*n_qubits]
        f_qubits = list(range(f_start, f_end))
        circ.initialize(f_norm, f_qubits)
        
        # Apply swap test between two n_q-qubit register pairs simultaneously
        # According to paper's circuit (a), we use one ancilla for both swap tests
        # First pair: |vec(K)⟩_first_nq and |u(θ)⟩
        # Second pair: |vec(K)⟩_second_nq and |f⟩
        
        # Hadamard on ancilla to create superposition
        circ.h(0)
        
        # Controlled-SWAP for first pair: |vec(K)⟩_first_nq and |u(θ)⟩
        for q1, q2 in zip(vec_K_first_qubits, u_theta_qubits):
            circ.cswap(0, q1, q2)
        
        # Controlled-SWAP for second pair: |vec(K)⟩_second_nq and |f⟩
        for q1, q2 in zip(vec_K_second_qubits, f_qubits):
            circ.cswap(0, q1, q2)
        
        # Final Hadamard on ancilla
        circ.h(0)
        
        return circ
    
    def build_denominator_circuit(
        self,
        vec_K: np.ndarray,
        vec_KT: np.ndarray,
        u_theta: np.ndarray
    ) -> QuantumCircuit:
        """
        Build circuit for computing CG denominator: ⟨u(θ)|K^T K|u(θ)⟩
        
        Uses vectorization: ⟨u(θ)|K^T K|u(θ)⟩ = ||K||² |⟨u(θ),vec(K^T)|vec(K),u(θ)⟩|²
        
        Qubit layout:
        - q[0]: ancilla for swap test
        - q[1 : 2*n_qubits+1]: |u(θ)⟩ ⊗ |vec(K^T)⟩
        - q[2*n_qubits+1 : 4*n_qubits+1]: |vec(K)⟩ ⊗ |u(θ)⟩
        
        Args:
            vec_K: Normalized vectorized matrix |vec(K)⟩
            vec_KT: Normalized vectorized transpose |vec(K^T)⟩
            u_theta: Pre-computed |u(θ)⟩ state vector
        
        Returns:
            Quantum circuit for CG denominator (for statevector simulation)
        """
        num_qubits = self.num_qubits_denominator
        circ = QuantumCircuit(num_qubits)
        
        # Calculate qubit ranges
        # |u(θ)⟩ ⊗ |vec(K^T)⟩ needs n_qubits + 2*n_qubits = 3*n_qubits qubits
        # |vec(K)⟩ ⊗ |u(θ)⟩ needs 2*n_qubits + n_qubits = 3*n_qubits qubits
        u_vecKT_start = 1
        u_vecKT_end = 1 + 3 * self.n_qubits
        vecK_u_start = u_vecKT_end
        vecK_u_end = num_qubits
        
        # Form tensor products classically
        u_theta_vecKT = np.kron(u_theta, vec_KT)
        vecK_u_theta = np.kron(vec_K, u_theta)
        
        # Left side: |u(θ)⟩ ⊗ |vec(K^T)⟩
        u_vecKT_qubits = list(range(u_vecKT_start, u_vecKT_end))
        circ.initialize(u_theta_vecKT, u_vecKT_qubits)
        
        # Right side: |vec(K)⟩ ⊗ |u(θ)⟩
        vecK_u_qubits = list(range(vecK_u_start, vecK_u_end))
        circ.initialize(vecK_u_theta, vecK_u_qubits)
        
        # Apply swap test between the two registers
        circ = apply_swap_test(
            circ,
            ancilla=0,
            reg1_qubits=u_vecKT_qubits,
            reg2_qubits=vecK_u_qubits
        )
        
        return circ

