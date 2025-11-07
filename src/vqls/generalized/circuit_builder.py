"""
Quantum circuit builder for DF-VQLS

- Matches Fig. 5(a),(c) in the DF-VQLS paper.

- Numerator: |⟨f|K|u(θ)⟩|^2 = ||K||^2 * |⟨vec(K)| u(θ)⊗f ⟩|^2

- Denominator: ⟨u|K^T K|u⟩ = ||K||^2 * |⟨ u⊗vec(K^T) | vec(K)⊗u ⟩|^2
"""

from typing import List

import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister

from .ansatz import Ansatz

from .utils import calculate_qubits


def apply_swap_test(
    circ: QuantumCircuit,
    ancilla: int,
    reg1_qubits: List[int],
    reg2_qubits: List[int]
) -> QuantumCircuit:
    """Standard swap test on two equal-length registers."""
    assert len(reg1_qubits) == len(reg2_qubits), "Register sizes must match"
    
    circ.h(ancilla)
    for q1, q2 in zip(reg1_qubits, reg2_qubits):
        circ.cswap(ancilla, q1, q2)
    circ.h(ancilla)
    
    return circ


def _assert_norm(x: np.ndarray, name: str, tol: float = 1e-9):
    n = np.linalg.norm(x)
    assert np.isfinite(n) and np.isclose(n, 1.0, atol=tol), f"{name} must be L2-normalized"


class CircuitBuilder:
    """
    Builds DF-VQLS circuits for the global/local cost components.

    Layout conventions (Qiskit little-endian respected via explicit qubit lists):
      - ancilla at q[0]
      - Registers are contiguous blocks; ordering is documented per method.
    """
    
    def __init__(self, matrix_size: int, ansatz: Ansatz):
        assert matrix_size > 0 and (matrix_size & (matrix_size - 1) == 0), \
            "matrix_size must be a power of 2"
        
        self.matrix_size = matrix_size
        self.n_qubits = calculate_qubits(matrix_size)  # nq = log2(N)
        self.ansatz = ansatz
        
        # Qubit counts (Fig. 5 in paper):
        # Numerator: 1 (anc) + 2*nq (vecK) + nq (u) + nq (f) = 4*nq + 1
        # Denominator: 1 (anc) + [nq + 2*nq + 2*nq + nq] = 6*nq + 1
        self.num_qubits_numerator = 4 * self.n_qubits + 1
        self.num_qubits_denominator = 6 * self.n_qubits + 1
    
    # ---------- Public builders ----------
    
    def build_numerator_circuit(
        self,
        vec_K: np.ndarray,     # length N^2, normalized (||vec(K)||=1)
        f_state: np.ndarray,   # length N, normalized
        u_theta: np.ndarray,   # length N, normalized (ansatz-prepared statevector)
        *,
        measure_ancilla: bool = False
    ) -> QuantumCircuit:
        """
        Circuit for |⟨f|K|u(θ)⟩|^2 via swap test:
          Prepare |ψ1⟩ = |vec(K)⟩ on 2*nq qubits
          Prepare |ψ2⟩ = |u(θ)⟩ ⊗ |f⟩ on nq + nq qubits
          Apply a single ancilla-controlled SWAP over two nq-sized pairs simultaneously:
            (vecK_first <-> u), (vecK_second <-> f)

        Qubit layout (total 4*nq+1):
          q[0]                  : ancilla
          q[1 : 1+nq)           : vecK_first (size nq)
          q[1+nq : 1+2nq)       : vecK_second (size nq)
          q[1+2nq : 1+3nq)      : u(θ)
          q[1+3nq : 1+4nq)      : f
        """
        N = 2 ** self.n_qubits
        
        assert vec_K.shape == (N * N,), "vec_K must be length N^2"
        assert u_theta.shape == (N,), "u_theta must be length N"
        assert f_state.shape == (N,), "f_state must be length N"
        
        _assert_norm(vec_K, "vec_K")
        _assert_norm(u_theta, "u_theta")
        _assert_norm(f_state, "f_state")
        
        nq = self.n_qubits
        num_qubits = self.num_qubits_numerator
        circ = QuantumCircuit(num_qubits)
        
        anc = 0
        vecK_first = list(range(1, 1 + nq))
        vecK_second = list(range(1 + nq, 1 + 2 * nq))
        u_regs = list(range(1 + 2 * nq, 1 + 3 * nq))
        f_regs = list(range(1 + 3 * nq, 1 + 4 * nq))
        
        # Prepare |vec(K)⟩ on 2*nq qubits
        circ.initialize(vec_K, vecK_first + vecK_second)
        
        # Prepare |u(θ)⟩ and |f⟩
        circ.initialize(u_theta, u_regs)
        circ.initialize(f_state, f_regs)
        
        # Ancilla-controlled SWAPs for two nq-pairs (matches Fig. 5(a))
        circ.h(anc)
        for q1, q2 in zip(vecK_first, u_regs):
            circ.cswap(anc, q1, q2)
        for q1, q2 in zip(vecK_second, f_regs):
            circ.cswap(anc, q1, q2)
        circ.h(anc)
        
        if measure_ancilla:
            c = ClassicalRegister(1, "c_anc")
            circ.add_register(c)
            circ.measure(anc, c[0])
        
        return circ
    
    def build_denominator_circuit(
        self,
        vec_K: np.ndarray,     # length N^2, normalized (||vec(K)||=1)
        vec_KT: np.ndarray,    # length N^2, normalized (||vec(K^T)||=1) = ||vec(K)||
        u_theta: np.ndarray,   # length N, normalized
        *,
        measure_ancilla: bool = False
    ) -> QuantumCircuit:
        """
        Circuit for ⟨u|K^T K|u⟩ via swap test on |u, vec(K^T)⟩ and |vec(K), u⟩.

        Qubit layout (total 6*nq+1) consistent with Fig. 5(c):
          q[0]                      : ancilla
          q[1 : 1+nq)               : u_ψ1
          q[1+nq : 1+2nq)           : vecKT_first
          q[1+2nq : 1+3nq)          : vecKT_second
          q[1+3nq : 1+4nq)          : vecK_first
          q[1+4nq : 1+5nq)          : vecK_second
          q[1+5nq : 1+6nq)          : u_ψ2

        CSWAP (three nq-pairs; global SWAP of the 3-block product):
          (u_ψ1 <-> vecK_first), (vecKT_first <-> u_ψ2), (vecKT_second <-> vecK_second)
        """
        N = 2 ** self.n_qubits
        
        assert vec_K.shape == (N * N,), "vec_K must be length N^2"
        assert vec_KT.shape == (N * N,), "vec_KT must be length N^2"
        assert u_theta.shape == (N,), "u_theta must be length N"
        
        _assert_norm(vec_K, "vec_K")
        _assert_norm(vec_KT, "vec_KT")
        _assert_norm(u_theta, "u_theta")
        
        nq = self.n_qubits
        num_qubits = self.num_qubits_denominator
        circ = QuantumCircuit(num_qubits)
        
        anc = 0
        u_psi1 = list(range(1, 1 + nq))
        vecKT_first = list(range(1 + nq, 1 + 2 * nq))
        vecKT_second = list(range(1 + 2 * nq, 1 + 3 * nq))
        vecK_first = list(range(1 + 3 * nq, 1 + 4 * nq))
        vecK_second = list(range(1 + 4 * nq, 1 + 5 * nq))
        u_psi2 = list(range(1 + 5 * nq, 1 + 6 * nq))
        
        # Prepare states
        circ.initialize(u_theta, u_psi1)
        circ.initialize(vec_KT, vecKT_first + vecKT_second)
        circ.initialize(vec_K,  vecK_first + vecK_second)
        circ.initialize(u_theta, u_psi2)
        
        # Ancilla-controlled SWAPs for THREE nq-pairs (matches Fig. 5(c))
        circ.h(anc)
        for q1, q2 in zip(u_psi1, vecK_first):
            circ.cswap(anc, q1, q2)
        for q1, q2 in zip(vecKT_first, u_psi2):
            circ.cswap(anc, q1, q2)
        # ★ 3rd pair (previously missing)
        for q1, q2 in zip(vecKT_second, vecK_second):
            circ.cswap(anc, q1, q2)
        circ.h(anc)
        
        if measure_ancilla:
            c = ClassicalRegister(1, "c_anc")
            circ.add_register(c)
            circ.measure(anc, c[0])
        
        return circ
    
    # ---------- Small helpers ----------
    
    @staticmethod
    def p0_from_counts(counts: dict, shots: int) -> float:
        """Return P(ancilla=0) from counts dict."""
        return counts.get("0", 0) / shots
    
    @staticmethod
    def overlap_sq_from_p0(p0: float) -> float:
        """Given P(0), return |⟨ψ1|ψ2⟩|^2 = 2P(0) - 1."""
        return max(0.0, min(1.0, 2.0 * p0 - 1.0))
