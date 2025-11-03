"""
Decomposition-Free Variational Quantum Linear Solver (DF-VQLS)

This module implements the DF-VQLS algorithm that eliminates the need for matrix
decomposition by using vectorization techniques and swap test circuits.

Key innovations:
- Vectorization techniques to transform matrix operations to inner products
- Swap test instead of Hadamard test
- Only 2 circuit executions per cost function evaluation (vs O(k²) in original VQLS)

References:
    Yongchun Xu and Heng Hu, "Decomposition-free variational quantum linear solver:
    Application in computational mechanics" (2025)
"""

from typing import List, Tuple, Union
import numpy as np
import random
from scipy.optimize import minimize, OptimizeResult
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ============================================================================
# 1. Matrix Vectorization Utilities
# ============================================================================

def vectorize_matrix(K: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert matrix K to normalized quantum state |vec(K)⟩.

    Implements vectorization by stacking columns (Fortran order) and normalizing
    by the Frobenius norm: |vec(K)⟩ = vec(K) / ||K||_F

    Args:
        K: Input matrix (N×N)

    Returns:
        Tuple of (normalized_vector, frobenius_norm)
    """
    # Flatten in column-major order (Fortran style)
    vec_K = K.flatten('F')

    # Compute Frobenius norm
    norm_K = np.linalg.norm(K, 'fro')

    # Normalize
    vec_K_normalized = vec_K / norm_K

    return vec_K_normalized, norm_K


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    return v / np.linalg.norm(v)


# ============================================================================
# 2. Hardware-Efficient Ansatz
# ============================================================================

def apply_hardware_efficient_ansatz(
    circ: QuantumCircuit,
    qubits: List[int],
    parameters: np.ndarray,
    num_layers: int = 3
) -> QuantumCircuit:
    """
    Apply hardware-efficient ansatz with RY rotations and CZ entanglement.

    The ansatz consists of multiple layers, each with:
    - Parameterized RY rotation on each qubit
    - CZ entanglement gates in a fixed pattern

    Args:
        circ: Quantum circuit to apply ansatz to
        qubits: List of qubit indices (should be 3 qubits for N=8)
        parameters: Flattened parameter array (num_qubits × num_layers)
        num_layers: Number of ansatz layers

    Returns:
        Modified quantum circuit with ansatz applied
    """
    num_qubits = len(qubits)
    params_reshaped = parameters.reshape(num_layers, num_qubits)

    for layer in range(num_layers):
        # RY rotations
        for i, qubit in enumerate(qubits):
            circ.ry(params_reshaped[layer, i], qubit)

        # CZ entanglement (skip on last layer)
        if layer < num_layers - 1:
            # Pattern: linear + circular
            circ.cz(qubits[0], qubits[1])
            circ.cz(qubits[1], qubits[2])
            if num_qubits > 3:
                circ.cz(qubits[2], qubits[0])

    return circ


# ============================================================================
# 3. Swap Test Circuit
# ============================================================================

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


# ============================================================================
# 4. CG Numerator Circuit (13 qubits)
# ============================================================================

def build_CG_numerator_circuit(
    params: np.ndarray,
    vec_K_normalized: np.ndarray,
    f_normalized: np.ndarray,
    u_theta: np.ndarray,
    num_ansatz_layers: int = 3
) -> QuantumCircuit:
    """
    Build circuit for computing CG numerator: |⟨f|K|u(θ)⟩|²

    Uses vectorization: |⟨f|K|u(θ)⟩|² = ||K||² |⟨vec(K)|u(θ),f⟩|²

    Qubit layout (13 qubits total):
    - q[0]: ancilla for swap test
    - q[1-6]: |vec(K)⟩ (6 qubits encode 64-dim vector)
    - q[7-12]: |u(θ)⟩ ⊗ |f⟩ (6 qubits for tensor product)

    Args:
        params: Ansatz parameters (9 values for 3 qubits × 3 layers)
        vec_K_normalized: Normalized vectorized matrix
        f_normalized: Normalized right-hand side vector
        u_theta: Pre-computed |u(θ)⟩ state vector
        num_ansatz_layers: Number of layers in ansatz

    Returns:
        Quantum circuit for CG numerator (no measurement, for statevector)
    """
    circ = QuantumCircuit(13)

    # Prepare |vec(K)⟩ on qubits 1-6
    circ.initialize(vec_K_normalized, [1, 2, 3, 4, 5, 6])

    # Form tensor product |u(θ)⟩ ⊗ |f⟩ classically
    u_theta_f = np.kron(u_theta, f_normalized)

    # Prepare |u(θ)⟩ ⊗ |f⟩ on qubits 7-12 (joint initialization!)
    circ.initialize(u_theta_f, [7, 8, 9, 10, 11, 12])

    # Apply swap test between q[1-6] and q[7-12]
    # Left side: qubits 1-6 (|vec(K)⟩)
    # Right side: qubits 7-12 (|u(θ)⟩ ⊗ |f⟩)
    circ = apply_swap_test(
        circ,
        ancilla=0,
        reg1_qubits=[1, 2, 3, 4, 5, 6],
        reg2_qubits=[7, 8, 9, 10, 11, 12]
    )

    return circ


# ============================================================================
# 5. CG Denominator Circuit (19 qubits)
# ============================================================================

def build_CG_denominator_circuit(
    params: np.ndarray,
    vec_K_normalized: np.ndarray,
    vec_KT_normalized: np.ndarray,
    u_theta: np.ndarray,
    num_ansatz_layers: int = 3
) -> QuantumCircuit:
    """
    Build circuit for computing CG denominator: ⟨u(θ)|K^T K|u(θ)⟩

    Uses vectorization: ⟨u(θ)|K^T K|u(θ)⟩ = ||K||² |⟨u(θ),vec(K^T)|vec(K),u(θ)⟩|²

    Qubit layout (19 qubits total):
    - q[0]: ancilla for swap test
    - q[1-9]: |u(θ)⟩ ⊗ |vec(K^T)⟩ (9 qubits for tensor product)
    - q[10-18]: |vec(K)⟩ ⊗ |u(θ)⟩ (9 qubits for tensor product)

    Args:
        params: Ansatz parameters
        vec_K_normalized: Normalized vectorized matrix K
        vec_KT_normalized: Normalized vectorized matrix K^T
        u_theta: Pre-computed |u(θ)⟩ state vector
        num_ansatz_layers: Number of layers in ansatz

    Returns:
        Quantum circuit for CG denominator (no measurement, for statevector)
    """
    circ = QuantumCircuit(19)

    # Form tensor products classically
    u_theta_vecKT = np.kron(u_theta, vec_KT_normalized)
    vecK_u_theta = np.kron(vec_K_normalized, u_theta)

    # Left side: |u(θ)⟩ ⊗ |vec(K^T)⟩ on qubits 1-9
    circ.initialize(u_theta_vecKT, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Right side: |vec(K)⟩ ⊗ |u(θ)⟩ on qubits 10-18
    circ.initialize(vecK_u_theta, [10, 11, 12, 13, 14, 15, 16, 17, 18])

    # Apply swap test between 9-qubit registers
    circ = apply_swap_test(
        circ,
        ancilla=0,
        reg1_qubits=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        reg2_qubits=[10, 11, 12, 13, 14, 15, 16, 17, 18]
    )

    return circ


# ============================================================================
# 6. CG Cost Function Computation
# ============================================================================

def compute_CG_cost(
    params: np.ndarray,
    K: np.ndarray,
    f: np.ndarray,
    simulator: AerSimulator,
    shots: int = None,
    num_ansatz_layers: int = 3,
    verbose: bool = True
) -> float:
    """
    Compute the global cost function CG(θ) = 1 - |⟨f|K|u(θ)⟩|² / ⟨u(θ)|K^T K|u(θ)⟩

    This cost function measures how well the variational state |u(θ)⟩ approximates
    the solution to Ku = f. When CG = 0, we have found the exact solution.

    Uses statevector mode for exact simulation (no shot noise).

    Args:
        params: Ansatz parameters (flattened array)
        K: Coefficient matrix (8×8)
        f: Right-hand side vector (8×1)
        simulator: Qiskit Aer simulator instance (statevector mode)
        shots: Ignored (kept for backward compatibility)
        num_ansatz_layers: Number of layers in ansatz
        verbose: Whether to print cost during optimization

    Returns:
        Cost function value (0 means perfect solution)
    """
    # Vectorize and normalize matrices
    vec_K, norm_K = vectorize_matrix(K)
    vec_KT, _ = vectorize_matrix(K.T)
    f_normalized = normalize_vector(f)

    # First, get |u(θ)⟩ by running the ansatz
    temp_circ = QuantumCircuit(3)
    temp_circ = apply_hardware_efficient_ansatz(temp_circ, [0, 1, 2], params, num_ansatz_layers)
    temp_circ.save_statevector()
    temp_result = simulator.run(transpile(temp_circ, simulator)).result()
    u_theta = np.asarray(temp_result.get_statevector(temp_circ))

    # ===== Compute Numerator: |⟨f|K|u(θ)⟩|² =====
    circ_num = build_CG_numerator_circuit(
        params, vec_K, f_normalized, u_theta, num_ansatz_layers
    )
    circ_num.save_statevector()

    # Run circuit
    result_num = simulator.run(transpile(circ_num, simulator)).result()
    sv_num = np.asarray(result_num.get_statevector(circ_num))

    # Extract P(0) from statevector
    P0_num = 0
    for i in range(len(sv_num)):
        if i % 2 == 0:  # Even indices = ancilla (qubit 0) is |0⟩
            P0_num += abs(sv_num[i])**2

    # Calculate inner product squared: |⟨ψ₁|ψ₂⟩|² = 2P(0) - 1
    inner_product_squared_num = abs(2 * P0_num - 1)

    # Numerator needs |⟨vec(K)|u,f⟩|² (squared overlap)
    numerator = (norm_K ** 2) * inner_product_squared_num

    # ===== Compute Denominator: ⟨u(θ)|K^T K|u(θ)⟩ =====
    circ_den = build_CG_denominator_circuit(
        params, vec_K, vec_KT, u_theta, num_ansatz_layers
    )
    circ_den.save_statevector()

    # Run circuit
    result_den = simulator.run(transpile(circ_den, simulator)).result()
    sv_den = np.asarray(result_den.get_statevector(circ_den))

    # Extract P(0) from statevector
    P0_den = 0
    for i in range(len(sv_den)):
        if i % 2 == 0:
            P0_den += abs(sv_den[i])**2

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

    if verbose:
        print(f"Cost: {cost:.6f} | Num: {numerator:.6f} | Den: {denominator:.6f} | P0_num: {P0_num:.4f} | P0_den: {P0_den:.4f}")

    return cost


# ============================================================================
# 7. Solution Extraction
# ============================================================================

def extract_solution_state(
    optimal_params: np.ndarray,
    num_ansatz_layers: int = 3
) -> np.ndarray:
    """
    Extract the quantum state |u(θ*)⟩ from optimal parameters.

    Args:
        optimal_params: Optimized ansatz parameters
        num_ansatz_layers: Number of layers in ansatz

    Returns:
        8-dimensional quantum state vector
    """
    # Create circuit with optimal parameters
    circ = QuantumCircuit(3)
    circ = apply_hardware_efficient_ansatz(circ, [0, 1, 2], optimal_params, num_ansatz_layers)
    circ.save_statevector()

    # Simulate to get statevector
    simulator = AerSimulator(method='statevector')
    transpiled = transpile(circ, simulator)
    job = simulator.run(transpiled)
    result = job.result()
    statevector = result.get_statevector(circ)

    # Convert to numpy array (real part only for real linear systems)
    u_quantum = np.real(np.array(statevector))

    return u_quantum


def scale_solution(
    u_quantum: np.ndarray,
    K: np.ndarray,
    f: np.ndarray
) -> np.ndarray:
    """
    Scale the quantum solution to recover true solution magnitude.

    The quantum state |u(θ*)⟩ is normalized, so we need to find the scale
    factor s that minimizes ||f - sK|u(θ*)⟩||² using least squares.

    Args:
        u_quantum: Normalized quantum solution state
        K: Coefficient matrix
        f: Right-hand side vector

    Returns:
        Scaled solution vector u = s|u(θ*)⟩
    """
    # Compute K|u⟩
    Ku = K @ u_quantum

    # Find optimal scale: s = (f^T Ku) / (Ku^T Ku)
    s = np.dot(f, Ku) / (np.dot(Ku, Ku) + 1e-10)

    # Scale solution
    u_scaled = s * u_quantum

    return u_scaled


# ============================================================================
# 8. Main Solver Function
# ============================================================================

def solve_dfvqls_8x8(
    K: np.ndarray,
    f: np.ndarray,
    optimizer: str = 'COBYLA',
    max_iter: int = 200,
    num_ansatz_layers: int = 3,
    shots: int = 8192,
    random_seed: Union[int, None] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, OptimizeResult]:
    """
    Solve the linear system Ku = f for an 8×8 matrix using DF-VQLS with CG cost function.

    This is the main entry point for the DF-VQLS algorithm. It uses the global
    cost function (CG) and requires no matrix decomposition.

    Args:
        K: 8×8 coefficient matrix
        f: 8×1 right-hand side vector
        optimizer: Optimization method ('COBYLA' or 'BFGS')
        max_iter: Maximum number of optimization iterations
        num_ansatz_layers: Number of layers in the variational ansatz
        shots: Number of measurement shots per circuit execution
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Tuple of (solution_vector, optimization_result)
    """
    # Validate inputs
    assert K.shape == (8, 8), "K must be 8×8 matrix"
    assert f.shape == (8,) or f.shape == (8, 1), "f must be 8-dimensional vector"
    f = f.flatten()  # Ensure f is 1D

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Create simulator once and reuse
    simulator = AerSimulator(method='statevector')

    # Initialize random parameters
    # 3 qubits × num_layers parameters
    num_params = 3 * num_ansatz_layers
    initial_params = np.random.uniform(0, 2 * np.pi, num_params)

    if verbose:
        print("=" * 70)
        print("DF-VQLS for 8×8 System (Statevector Mode - Exact)")
        print("=" * 70)
        print(f"Matrix size: 8×8")
        print(f"Optimizer: {optimizer}")
        print(f"Max iterations: {max_iter}")
        print(f"Ansatz layers: {num_ansatz_layers}")
        print(f"Simulation mode: Statevector (no shot noise)")
        print(f"Initial parameters: {num_params}")
        print("=" * 70)
        print("\nStarting optimization...\n")

    # Run optimization
    result = minimize(
        fun=lambda p: compute_CG_cost(
            p, K, f, simulator, shots, num_ansatz_layers, verbose
        ),
        x0=initial_params,
        method=optimizer,
        options={'maxiter': max_iter, 'disp': verbose}
    )

    if verbose:
        print("\n" + "=" * 70)
        print("Optimization Results:")
        print("=" * 70)
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nfev}")
        print(f"Final cost: {result.fun:.6f}")

    # Extract and scale solution
    u_quantum = extract_solution_state(result.x, num_ansatz_layers)
    u_solution = scale_solution(u_quantum, K, f)

    if verbose:
        print("\nSolution extracted and scaled.")
        print("=" * 70)

    return u_solution, result


# ============================================================================
# 9. Testing and Validation
# ============================================================================

def test_dfvqls_example():
    """
    Test DF-VQLS with the example from the paper (tridiagonal system).

    Equation 37 from paper:
    K is a tridiagonal matrix with 2 on diagonal and -1 on off-diagonals
    f = [2, 0, 2, 5, 0, 0, 0, 0]^T
    """
    print("\n" + "=" * 70)
    print("Testing DF-VQLS with Paper Example (8×8 Tridiagonal System)")
    print("=" * 70)

    # Define the linear system from paper
    K = np.array([
        [ 2, -1,  0,  0,  0,  0,  0,  0],
        [-1,  2, -1,  0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0,  0,  0,  0],
        [ 0,  0, -1,  2, -1,  0,  0,  0],
        [ 0,  0,  0, -1,  2, -1,  0,  0],
        [ 0,  0,  0,  0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0, -1,  2, -1],
        [ 0,  0,  0,  0,  0,  0, -1,  2]
    ], dtype=float)

    f = np.array([2, 0, 2, 5, 0, 0, 0, 0], dtype=float)

    # Solve with classical method for reference
    u_classical = np.linalg.solve(K, f)

    print("\nClassical solution:")
    print(u_classical)

    # Solve with DF-VQLS
    u_quantum, result = solve_dfvqls_8x8(
        K, f,
        optimizer='COBYLA',
        max_iter=200,
        num_ansatz_layers=3,
        shots=8192,
        random_seed=42,
        verbose=True
    )

    print("\nDF-VQLS solution:")
    print(u_quantum)

    # Compute error
    relative_error = np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical)

    print("\n" + "=" * 70)
    print("Comparison with Classical Solution:")
    print("=" * 70)
    print(f"Relative Error: {relative_error:.6e}")
    print(f"Target Accuracy: ~1e-4 to 1e-5 (from paper)")

    if relative_error < 1e-3:
        print("✓ Excellent accuracy achieved!")
    elif relative_error < 1e-2:
        print("✓ Good accuracy achieved!")
    else:
        print("⚠ Moderate accuracy - consider more iterations or layers")

    print("=" * 70)


def main():
    """Main function to run DF-VQLS tests."""
    test_dfvqls_example()


if __name__ == "__main__":
    main()
