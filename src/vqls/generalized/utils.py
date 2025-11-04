"""
Utility functions for DF-VQLS

Provides matrix vectorization, size validation, and other helper functions.
"""

import math
import numpy as np
from typing import Tuple


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

