"""
DF-VQLS implementation for 4×4 systems (faster for testing)
"""

from typing import Tuple, Union
import numpy as np
import random
from scipy.optimize import minimize, OptimizeResult
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def vectorize_matrix(K: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert matrix K to normalized |vec(K)⟩"""
    vec_K = K.flatten('F')
    norm_K = np.linalg.norm(K, 'fro')
    return vec_K / norm_K, norm_K


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector"""
    return v / np.linalg.norm(v)


def apply_ansatz_4x4(circ, qubits, params, num_layers=2):
    """Apply ansatz for 2 qubits (4×4 system)"""
    params_reshaped = params.reshape(num_layers, 2)
    for layer in range(num_layers):
        circ.ry(params_reshaped[layer, 0], qubits[0])
        circ.ry(params_reshaped[layer, 1], qubits[1])
        if layer < num_layers - 1:
            circ.cz(qubits[0], qubits[1])
    return circ


def apply_swap_test(circ, ancilla, reg1, reg2):
    """Apply swap test"""
    assert len(reg1) == len(reg2)
    circ.h(ancilla)
    for q1, q2 in zip(reg1, reg2):
        circ.cswap(ancilla, q1, q2)
    circ.h(ancilla)
    return circ


def build_CG_numerator_4x4(params, vec_K, f_norm, num_layers=2):
    """
    CG numerator for 4×4 system
    9 qubits: 1 ancilla + 4 (vec(K)) + 2 (u(θ)) + 2 (f)
    """
    circ = QuantumCircuit(9, 1)

    # Prepare |vec(K)⟩ on qubits 1-4
    circ.initialize(vec_K, [1, 2, 3, 4])

    # Prepare |u(θ)⟩ on qubits 5-6
    circ = apply_ansatz_4x4(circ, [5, 6], params, num_layers)

    # Prepare |f⟩ on qubits 7-8
    circ.initialize(f_norm, [7, 8])

    # Swap test: compare q[1-4] with q[5-8]
    circ = apply_swap_test(circ, 0, [1,2,3,4], [5,6,7,8])

    circ.measure(0, 0)
    return circ


def build_CG_denominator_4x4(params, vec_K, vec_KT, num_layers=2):
    """
    CG denominator for 4×4 system
    13 qubits: 1 ancilla + 6 (u+vec(K^T)) + 6 (vec(K)+u)
    """
    circ = QuantumCircuit(13, 1)

    # Left: |u(θ)⟩ ⊗ |vec(K^T)⟩ on qubits 1-6
    circ = apply_ansatz_4x4(circ, [1, 2], params, num_layers)
    circ.initialize(vec_KT, [3, 4, 5, 6])

    # Right: |vec(K)⟩ ⊗ |u(θ)⟩ on qubits 7-12
    circ.initialize(vec_K, [7, 8, 9, 10])
    circ = apply_ansatz_4x4(circ, [11, 12], params, num_layers)

    # Swap test between 6-qubit registers
    circ = apply_swap_test(circ, 0, [1,2,3,4,5,6], [7,8,9,10,11,12])

    circ.measure(0, 0)
    return circ


def compute_CG_cost_4x4(params, K, f, simulator, shots=None, num_layers=2, verbose=True):
    """Compute CG cost for 4×4 system using statevector (exact)"""
    vec_K, norm_K = vectorize_matrix(K)
    vec_KT, _ = vectorize_matrix(K.T)
    f_norm = normalize_vector(f)

    # Numerator - prepare states correctly
    circ_num = QuantumCircuit(9)

    # Left side: |vec(K)⟩ on qubits 1-4
    circ_num.initialize(vec_K, [1, 2, 3, 4])

    # Right side: |u(θ)⟩ ⊗ |f⟩ on qubits 5-8
    # First get |u(θ)⟩ by simulating the ansatz
    temp_circ = QuantumCircuit(2)
    temp_circ = apply_ansatz_4x4(temp_circ, [0, 1], params, num_layers)
    temp_circ.save_statevector()
    temp_result = simulator.run(transpile(temp_circ, simulator)).result()
    u_theta = np.asarray(temp_result.get_statevector(temp_circ))

    # Form tensor product: |u(θ)⟩ ⊗ |f⟩
    u_theta_f = np.kron(u_theta, f_norm)
    circ_num.initialize(u_theta_f, [5, 6, 7, 8])

    # Apply swap test
    circ_num = apply_swap_test(circ_num, 0, [1,2,3,4], [5,6,7,8])
    circ_num.save_statevector()

    result_num = simulator.run(transpile(circ_num, simulator)).result()
    sv_num = np.asarray(result_num.get_statevector(circ_num))

    # P(0) is sum of probabilities where ancilla (qubit 0) is |0⟩
    # For 9 qubits, ancilla is LSB in statevector indexing
    P0_num = 0
    for i in range(len(sv_num)):
        if i % 2 == 0:  # Even indices = ancilla is |0⟩
            P0_num += abs(sv_num[i])**2

    inner_product_sq_num = abs(2 * P0_num - 1)
    # Numerator needs |⟨vec(K)|u,f⟩|² - squared overlap is correct
    numerator = (norm_K ** 2) * inner_product_sq_num

    # Denominator - prepare states correctly
    circ_den = QuantumCircuit(13)

    # Get |u(θ)⟩ (already computed above, reuse)
    # Form left side: |u(θ)⟩ ⊗ |vec(K^T)⟩
    u_theta_vecKT = np.kron(u_theta, vec_KT)
    circ_den.initialize(u_theta_vecKT, [1, 2, 3, 4, 5, 6])

    # Form right side: |vec(K)⟩ ⊗ |u(θ)⟩
    vecK_u_theta = np.kron(vec_K, u_theta)
    circ_den.initialize(vecK_u_theta, [7, 8, 9, 10, 11, 12])

    # Apply swap test
    circ_den = apply_swap_test(circ_den, 0, [1,2,3,4,5,6], [7,8,9,10,11,12])
    circ_den.save_statevector()

    result_den = simulator.run(transpile(circ_den, simulator)).result()
    sv_den = np.asarray(result_den.get_statevector(circ_den))

    # P(0) for denominator
    P0_den = 0
    for i in range(len(sv_den)):
        if i % 2 == 0:
            P0_den += abs(sv_den[i])**2

    inner_product_sq_den = abs(2 * P0_den - 1)
    # Denominator needs ⟨u|K^T K|u⟩ = ⟨u⊗vec(K^T)|vec(K)⊗u⟩ (inner product, NOT squared!)
    # Swap test gives |⟨ψ₁|ψ₂⟩|², so take sqrt to get the inner product value
    inner_product_den = np.sqrt(inner_product_sq_den)
    denominator = (norm_K ** 2) * inner_product_den

    cost = 1.0 - (numerator / (denominator + 1e-10))

    if verbose:
        print(f"Cost: {cost:.6f} | Num: {numerator:.6f} | Den: {denominator:.6f} | P0_num: {P0_num:.4f} | P0_den: {P0_den:.4f}")

    return cost


def extract_solution_4x4(params, num_layers=2):
    """Extract solution state"""
    circ = QuantumCircuit(2)
    circ = apply_ansatz_4x4(circ, [0, 1], params, num_layers)
    circ.save_statevector()

    sim = AerSimulator(method='statevector')
    result = sim.run(transpile(circ, sim)).result()
    return np.real(np.array(result.get_statevector(circ)))


def scale_solution(u_quantum, K, f):
    """Scale solution"""
    Ku = K @ u_quantum
    s = np.dot(f, Ku) / (np.dot(Ku, Ku) + 1e-10)
    return s * u_quantum


def solve_dfvqls_4x4(K, f, max_iter=100, num_layers=2, seed=42, verbose=True):
    """Solve 4×4 system with DF-VQLS using statevector (exact)"""
    assert K.shape == (4, 4), "K must be 4×4"
    assert f.shape == (4,) or f.shape == (4, 1), "f must be 4D"
    f = f.flatten()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    simulator = AerSimulator(method='statevector')
    initial_params = np.random.uniform(0, 2*np.pi, 2*num_layers)  # 2 qubits × num_layers

    if verbose:
        print("="*70)
        print("DF-VQLS for 4×4 System (Statevector Mode - Exact)")
        print("="*70)
        print(f"Max iterations: {max_iter}, Layers: {num_layers}")
        print("="*70 + "\n")

    result = minimize(
        fun=lambda p: compute_CG_cost_4x4(p, K, f, simulator, None, num_layers, verbose),
        x0=initial_params,
        method='COBYLA',
        options={'maxiter': max_iter}
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimization complete: {result.nfev} iterations")
        print(f"Final cost: {result.fun:.6f}")
        print("="*70)

    u_quantum = extract_solution_4x4(result.x, num_layers)
    u_scaled = scale_solution(u_quantum, K, f)

    return u_scaled, result


def test_4x4():
    """Test with 4×4 tridiagonal system"""
    print("\nTesting DF-VQLS with 4×4 Tridiagonal System\n")

    K = np.array([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]
    ], dtype=float)

    f = np.array([1, 0, 0, 1], dtype=float)

    # Classical solution
    u_classical = np.linalg.solve(K, f)
    print("Classical solution:")
    print(u_classical)
    print()

    # Quantum solution
    u_quantum, result = solve_dfvqls_4x4(
        K, f,
        max_iter=200,  # Increased for better convergence
        num_layers=3,  # More expressive ansatz
        seed=42,
        verbose=True
    )

    print("\nDF-VQLS solution:")
    print(u_quantum)

    error = np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical)
    print(f"\n{'='*70}")
    print(f"Relative Error: {error:.6e}")

    if error < 0.1:
        print("✓ Excellent - algorithm working!")
    elif error < 0.5:
        print("✓ Good - may need more iterations")
    else:
        print("⚠ High error - needs tuning")
    print("="*70)


if __name__ == "__main__":
    test_4x4()
