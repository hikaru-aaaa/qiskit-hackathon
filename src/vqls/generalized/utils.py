"""
Utility functions for DF-VQLS

Provides matrix vectorization, size validation, and other helper functions.
"""

import math
import numpy as np
import warnings
from typing import Tuple, Optional


def validate_matrix_size(size: int) -> bool:
    """
    Validate that matrix size is a power of 2.
    
    Args:
        size: Matrix dimension (must be 2^n)
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if size <= 0:
        raise ValueError(f"Matrix size must be positive, got {size}")
    
    if not (size & (size - 1) == 0):  # Check if power of 2
        raise ValueError(
            f"Matrix size must be a power of 2 (e.g., 2, 4, 8, 16, 32), "
            f"got {size}"
        )
    
    return True


def calculate_qubits(matrix_size: int) -> int:
    """
    Calculate number of qubits needed for matrix size.
    
    Args:
        matrix_size: Matrix dimension (must be power of 2)
    
    Returns:
        Number of qubits needed: log2(matrix_size)
    
    Raises:
        ValueError: If matrix_size is not a power of 2
    """
    validate_matrix_size(matrix_size)
    return int(math.log2(matrix_size))


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
    
    # Avoid division by zero
    if norm_K < 1e-10:
        raise ValueError("Matrix K has zero Frobenius norm")
    
    # Normalize
    vec_K_normalized = vec_K / norm_K
    
    return vec_K_normalized, norm_K


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        v: Input vector
    
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError("Vector has zero norm")
    return v / norm


def calculate_num_parameters(num_qubits: int, num_layers: int) -> int:
    """
    Calculate number of variational parameters needed.
    
    Args:
        num_qubits: Number of qubits
        num_layers: Number of ansatz layers
    
    Returns:
        Total number of parameters: num_qubits × num_layers
    """
    return num_qubits * num_layers


def scale_solution(
    u_quantum: np.ndarray,
    K: np.ndarray,
    f: np.ndarray
) -> np.ndarray:
    """
    Scale the quantum solution to match the classical solution scale.

    The quantum state |u(θ)⟩ is normalized, so we need to find a scale
    factor s such that s|u(θ)⟩ approximates the true solution.

    Uses least squares: s = ⟨f|K|u⟩ / ⟨u|K^T K|u⟩

    Args:
        u_quantum: Normalized quantum solution state
        K: Coefficient matrix
        f: Right-hand side vector

    Returns:
        Scaled solution vector
    """
    # Compute K|u⟩
    Ku = K @ u_quantum

    # Scale factor using least squares
    numerator = np.dot(f, Ku)
    denominator = np.dot(Ku, Ku)

    if denominator < 1e-10:
        raise ValueError("Denominator too small for scaling")

    s = numerator / denominator

    return s * u_quantum


def qsvt_to_theta_initialization(
    x_qsvt: np.ndarray,
    ansatz,
    num_qubits: int,
    max_iter: int = 50,
    verbose: bool = False
) -> np.ndarray:
    """
    Find ansatz parameters that best approximate QSVT solution.

    This function solves the optimization problem:
        θ_init = argmax_θ |⟨u(θ)|x_qsvt⟩|²

    where |u(θ)⟩ = V(θ)|0⟩ is the ansatz-generated state. The resulting
    parameters provide a warm start for DF-VQLS optimization, reducing
    the number of iterations needed to converge.

    Args:
        x_qsvt: QSVT solution vector (can be unnormalized)
        ansatz: Ansatz instance (e.g., HardwareEfficientAnsatz)
        num_qubits: Number of qubits in the system
        max_iter: Maximum number of optimization iterations (default: 50)
        verbose: Whether to print optimization progress

    Returns:
        θ_init: Optimized ansatz parameters that maximize overlap with x_qsvt

    Raises:
        ValueError: If x_qsvt has zero norm or invalid shape

    Notes:
        - Uses scipy.optimize.minimize with COBYLA method (derivative-free)
        - Typical overhead: 1-5 seconds for systems up to 8×8
        - Falls back to random initialization if optimization fails
        - The overlap |⟨u(θ)|x⟩|² ∈ [0, 1], where 1 means perfect match

    Example:
        >>> from src.vqls.generalized.ansatz import HardwareEfficientAnsatz
        >>> ansatz = HardwareEfficientAnsatz(num_qubits=2, num_layers=3)
        >>> x_qsvt = np.array([0.125, 0.625, 0.0, 0.0])  # QSVT solution
        >>> theta_init = qsvt_to_theta_initialization(
        ...     x_qsvt, ansatz, num_qubits=2, verbose=True
        ... )
        >>> # Use theta_init as initial parameters for DF-VQLS
    """
    from scipy.optimize import minimize
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    # Validate input
    if not isinstance(x_qsvt, np.ndarray):
        x_qsvt = np.array(x_qsvt)

    # Flatten if needed
    x_qsvt = x_qsvt.flatten()

    # Check norm
    x_norm = np.linalg.norm(x_qsvt)
    if x_norm < 1e-10:
        warnings.warn(
            "QSVT solution has near-zero norm. "
            "Using random initialization instead."
        )
        return np.random.uniform(0, 2 * np.pi, ansatz.num_parameters())

    # Normalize to quantum state
    x_target = x_qsvt / x_norm

    # Pad to 2^n_qubits if needed
    n_states = 2**num_qubits
    if len(x_target) < n_states:
        if verbose:
            print(f"  Padding x_qsvt from {len(x_target)} to {n_states} dimensions")
        x_padded = np.zeros(n_states, dtype=complex)
        x_padded[:len(x_target)] = x_target
        x_target = x_padded / np.linalg.norm(x_padded)
    elif len(x_target) > n_states:
        raise ValueError(
            f"x_qsvt has {len(x_target)} elements but only {n_states} states "
            f"are representable with {num_qubits} qubits"
        )

    # Convert to complex if needed (for inner product)
    x_target = x_target.astype(complex)

    def overlap_cost(theta):
        """
        Compute 1 - |⟨u(θ)|x_target⟩|² (cost to minimize).

        Lower cost → higher overlap → better initialization.
        """
        try:
            # Build circuit with current parameters
            qc = QuantumCircuit(num_qubits)
            ansatz.apply(qc, list(range(num_qubits)), theta)

            # Get statevector |u(θ)⟩
            u_theta = Statevector.from_instruction(qc).data

            # Compute overlap (fidelity)
            # Inner product: ⟨x|u⟩ = Σᵢ x*ᵢ uᵢ
            overlap = np.abs(np.vdot(x_target, u_theta))**2

            return 1.0 - overlap  # Minimize 1 - fidelity

        except Exception as e:
            if verbose:
                print(f"  Warning: Circuit execution failed - {e}")
            return 1.0  # Worst case (no overlap)

    # Initialize with random parameters
    theta_random = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters())

    # Optimize
    if verbose:
        print(f"  Optimizing {ansatz.num_parameters()} parameters "
              f"(max_iter={max_iter})...")

    result = minimize(
        overlap_cost,
        x0=theta_random,
        method='COBYLA',
        options={'maxiter': max_iter, 'disp': verbose}
    )

    # Report results
    final_overlap = 1.0 - result.fun
    if verbose:
        print(f"  ✓ Optimization complete:")
        print(f"    Final overlap: {final_overlap:.4f}")
        print(f"    Iterations: {result.nfev}")
        print(f"    Success: {result.success}")

    # Warn if poor overlap
    if final_overlap < 0.5:
        warnings.warn(
            f"Low overlap achieved ({final_overlap:.3f}). "
            f"QSVT initialization may not improve convergence."
        )

    return result.x


def recommend_initialization_strategy(
    qsvt_error: Optional[float] = None,
    matrix_size: int = 4,
    time_budget_seconds: Optional[float] = None
) -> str:
    """
    Recommend whether to use QSVT initialization based on problem characteristics.

    Args:
        qsvt_error: Relative error of QSVT solution (if available)
        matrix_size: Size of the matrix (N for N×N system)
        time_budget_seconds: Maximum time budget for solving (optional)

    Returns:
        One of: 'qsvt_init', 'random', 'skip_qsvt'
        - 'qsvt_init': Use QSVT solution to initialize DF-VQLS
        - 'random': Skip QSVT, use random DF-VQLS initialization
        - 'skip_qsvt': QSVT error is good enough, don't run DF-VQLS

    Example:
        >>> strategy = recommend_initialization_strategy(
        ...     qsvt_error=1e-4, matrix_size=2
        ... )
        >>> print(strategy)  # 'skip_qsvt' (QSVT is accurate enough)
    """
    # If QSVT is very accurate, use it directly
    if qsvt_error is not None and qsvt_error < 1e-3:
        return 'skip_qsvt'

    # If QSVT is moderately good, use it to initialize
    if qsvt_error is not None and qsvt_error < 0.1:
        return 'qsvt_init'

    # If time budget is tight and matrix is small, skip QSVT
    if time_budget_seconds is not None and time_budget_seconds < 30:
        if matrix_size <= 4:
            return 'random'

    # Default: use QSVT initialization if error is moderate
    if qsvt_error is not None and qsvt_error < 0.5:
        return 'qsvt_init'

    # If QSVT error is very poor or unknown, use random
    return 'random'

