"""
Quantum circuit builder for DF-VQLS

Constructs numerator and denominator circuits for cost function computation.
"""

from typing import List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from .ansatz import Ansatz
from .utils import calculate_qubits


def apply_swap_test(
    circ: QuantumCircuit, ancilla: int, reg1_qubits: List[int], reg2_qubits: List[int]
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
    assert len(reg1_qubits) == len(reg2_qubits), (
        "Register sizes must match for swap test"
    )

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
        self, vec_K: np.ndarray, f_norm: np.ndarray, u_theta: np.ndarray
    ) -> QuantumCircuit:
        """
        Build circuit for computing CG numerator: |⟨f|K|u(θ)⟩|²

        Uses vectorization: |⟨f|K|u(θ)⟩|² = ||K||² |⟨vec(K)|u(θ),f⟩|²

        Qubit layout:
        - q[0]: ancilla for swap test
        - q[1 : 2*n_qubits+1]: |vec(K)⟩
        - q[2*n_qubits+1 : 4*n_qubits+1]: |u(θ)⟩ ⊗ |f⟩

        Args:
            vec_K: Normalized vectorized matrix |vec(K)⟩
            f_norm: Normalized right-hand side vector |f⟩
            u_theta: Pre-computed |u(θ)⟩ state vector

        Returns:
            Quantum circuit for CG numerator (for statevector simulation)
        """
        num_qubits = self.num_qubits_numerator
        circ = QuantumCircuit(num_qubits)

        # Calculate qubit ranges
        vec_K_start = 1
        vec_K_end = 1 + 2 * self.n_qubits
        uf_start = vec_K_end
        uf_end = num_qubits

        # Prepare |vec(K)⟩ on qubits [1, 2*n_qubits]
        vec_K_qubits = list(range(vec_K_start, vec_K_end))
        circ.initialize(vec_K, vec_K_qubits)

        # Form tensor product |u(θ)⟩ ⊗ |f⟩ classically
        u_theta_f = np.kron(u_theta, f_norm)

        # Prepare |u(θ)⟩ ⊗ |f⟩ on qubits [2*n_qubits+1, 4*n_qubits]
        uf_qubits = list(range(uf_start, uf_end))
        circ.initialize(u_theta_f, uf_qubits)

        # Apply swap test between the two registers
        circ = apply_swap_test(
            circ, ancilla=0, reg1_qubits=vec_K_qubits, reg2_qubits=uf_qubits
        )

        return circ

    def build_numerator_circuit_quantum(
        self,
        vec_K: np.ndarray,
        f_norm: np.ndarray,
        u_theta_circuit: QuantumCircuit,  # <-- Takes a circuit
    ) -> QuantumCircuit:
        num_qubits = self.num_qubits_numerator
        circ = QuantumCircuit(num_qubits)

        # ... (Qubit range calculations) ...
        vec_K_qubits = list(range(1, 1 + 2 * self.n_qubits))
        u_qubits = list(range(1 + 2 * self.n_qubits, 1 + 3 * self.n_qubits))
        f_qubits = list(range(1 + 3 * self.n_qubits, num_qubits))
        uf_qubits = list(range(1 + 2 * self.n_qubits, num_qubits))

        # 1. Prepare |vec(K)⟩
        prep_K = StatePreparation(vec_K, label="vec(K)")
        circ.append(prep_K, vec_K_qubits)

        # 2. Append the |u(θ)⟩ circuit
        circ.append(u_theta_circuit, u_qubits)

        # 3. Prepare |f⟩
        prep_f = StatePreparation(f_norm, label="f")
        circ.append(prep_f, f_qubits)

        # 4. Apply swap test
        circ = apply_swap_test(
            circ, ancilla=0, reg1_qubits=vec_K_qubits, reg2_qubits=uf_qubits
        )
        return circ

    def build_denominator_circuit(
        self, vec_K: np.ndarray, vec_KT: np.ndarray, u_theta: np.ndarray
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
            circ, ancilla=0, reg1_qubits=u_vecKT_qubits, reg2_qubits=vecK_u_qubits
        )

        return circ

    def build_denominator_circuit_quantum(
        self,
        vec_K: np.ndarray,
        vec_KT: np.ndarray,
        u_theta_circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        num_qubits = self.num_qubits_denominator
        circ = QuantumCircuit(num_qubits)
        n = self.n_qubits

        # Register 1 (Left): |u(θ)⟩ ⊗ |vec(K^T)⟩
        u1_qubits = list(range(1, 1 + n))
        vecKT_qubits = list(range(1 + n, 1 + 3 * n))

        # Register 2 (Right): |vec(K)⟩ ⊗ |u(θ)⟩
        vecK_qubits = list(range(1 + 3 * n, 1 + 5 * n))
        u2_qubits = list(range(1 + 5 * n, num_qubits))  # num_qubits = 1 + 6*n

        # Define the full register ranges for the swap test
        reg1_qubits = list(range(1, 1 + 3 * n))
        reg2_qubits = list(range(1 + 3 * n, num_qubits))

        # --- Build Register 1: |u(θ)⟩ ⊗ |vec(K^T)⟩ ---

        # 1. Append |u(θ)⟩ circuit
        # We compose a copy to avoid issues if u_theta_circuit is used elsewhere
        circ.append(u_theta_circuit.copy(), u1_qubits)

        # 2. Prepare |vec(K^T)⟩
        prep_KT = StatePreparation(vec_KT, label="vec(K^T)")
        circ.append(prep_KT, vecKT_qubits)

        # --- Build Register 2: |vec(K)⟩ ⊗ |u(θ)⟩ ---

        # 3. Prepare |vec(K)⟩
        prep_K = StatePreparation(vec_K, label="vec(K)")
        circ.append(prep_K, vecK_qubits)

        # 4. Append |u(θ)⟩ circuit (a second time)
        circ.append(u_theta_circuit.copy(), u2_qubits)

        # --- Apply swap test between the two full registers ---
        circ = apply_swap_test(
            circ, ancilla=0, reg1_qubits=reg1_qubits, reg2_qubits=reg2_qubits
        )

        return circ
