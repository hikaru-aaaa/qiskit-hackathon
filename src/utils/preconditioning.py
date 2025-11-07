"""
Preconditioning utilities for improving DF-VQLS convergence.

Implements Jacobi preconditioning as described in:
"Decomposition-free variational quantum linear solver" (Xu & Hu, 2025)
Pages 22-23, Figure 13

Preconditioning reduces the condition number κ(K), which improves optimizer
convergence by preventing barren plateaus in the cost function landscape.
"""

import numpy as np
from typing import Tuple


def jacobi_precondition(
    A: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Jacobi preconditioning to linear system Ax=b.

    Preconditioning transforms the system to:
        A_pre * u_pre = b_pre
    where:
        A_pre = D^(-1/2) * A * D^(-1/2)
        b_pre = D^(-1/2) * b
        D = diag(A)

    This reduces the condition number κ(A), improving optimizer convergence.

    The mechanism:
    - High κ(K) causes K^T K to dominate the cost function denominator
    - This creates flat optimization landscapes (barren plateaus)
    - Preconditioning reduces κ → steeper gradients → better convergence

    Args:
        A: Coefficient matrix (N×N)
        b: Right-hand side vector (N,)

    Returns:
        A_pre: Preconditioned matrix (N×N)
        b_pre: Preconditioned RHS vector (N,)
        D_sqrt_inv: Diagonal scaling matrix D^(-1/2) (N×N)
                    Needed for solution recovery: u = D^(-1/2) * u_pre

    Example:
        >>> A = np.array([[4, 1], [1, 3]])
        >>> b = np.array([1, 2])
        >>> A_pre, b_pre, D_inv = jacobi_precondition(A, b)
        >>> # Solve preconditioned system...
        >>> u = recover_solution(u_pre, D_inv)

    References:
        Xu & Hu (2025), Equation 45, pages 22-23
        Figure 13 shows κ reduction: 2.95×10⁶ → 4.29×10²
    """
    # Extract diagonal of A
    D_diag = np.diag(A)

    # Handle zero or near-zero diagonal elements
    if np.any(np.abs(D_diag) < 1e-12):
        raise ValueError(
            "Matrix has zero or near-zero diagonal elements. "
            "Jacobi preconditioning requires non-zero diagonal."
        )

    # Compute D^(-1/2)
    D_sqrt_inv_diag = 1.0 / np.sqrt(D_diag)
    D_sqrt_inv = np.diag(D_sqrt_inv_diag)

    # Apply preconditioning
    A_pre = D_sqrt_inv @ A @ D_sqrt_inv
    b_pre = D_sqrt_inv @ b

    return A_pre, b_pre, D_sqrt_inv


def recover_solution(u_pre: np.ndarray, D_sqrt_inv: np.ndarray) -> np.ndarray:
    """
    Recover original solution from preconditioned solution.

    Given the solution u_pre to the preconditioned system, compute:
        u = D^(-1/2) * u_pre

    Args:
        u_pre: Solution to preconditioned system (N,)
        D_sqrt_inv: Diagonal scaling matrix from jacobi_precondition()

    Returns:
        u: Solution to original system (N,)

    Example:
        >>> A_pre, b_pre, D_inv = jacobi_precondition(A, b)
        >>> u_pre = solve(A_pre, b_pre)  # Solve preconditioned system
        >>> u = recover_solution(u_pre, D_inv)  # Recover original solution
    """
    return D_sqrt_inv @ u_pre


def calculate_condition_number(A: np.ndarray) -> float:
    """
    Calculate condition number κ(A) = σ_max / σ_min.

    The condition number measures how sensitive the solution is to perturbations.
    High κ(A) >> 1 indicates ill-conditioned system, leading to:
    - Poor optimizer convergence (barren plateaus)
    - K^T K term dominates cost function denominator
    - Flat optimization landscape with vanishing gradients

    Args:
        A: Matrix (N×N)

    Returns:
        κ(A): Condition number (positive float)

    Example:
        >>> A = create_matrix_with_condition_number(8, kappa=4)
        >>> kappa = calculate_condition_number(A)
        >>> print(f"Condition number: {kappa:.2e}")
        Condition number: 2.95e+06
        >>> # This high κ will cause convergence issues!
    """
    return np.linalg.cond(A)


def print_condition_number_analysis(
    A_original: np.ndarray,
    A_preconditioned: np.ndarray,
    matrix_name: str = "K"
) -> None:
    """
    Print condition number comparison before/after preconditioning.

    Args:
        A_original: Original coefficient matrix
        A_preconditioned: Preconditioned matrix
        matrix_name: Name to display (default: "K")
    """
    kappa_orig = calculate_condition_number(A_original)
    kappa_pre = calculate_condition_number(A_preconditioned)
    improvement = kappa_orig / kappa_pre

    print(f"\n{'='*60}")
    print(f"Condition Number Analysis")
    print(f"{'='*60}")
    print(f"Original κ({matrix_name}):        {kappa_orig:.2e}")
    print(f"Preconditioned κ({matrix_name}):  {kappa_pre:.2e}")
    print(f"Improvement factor:      {improvement:.2f}x")
    print(f"{'='*60}\n")
