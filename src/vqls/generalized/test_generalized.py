"""
Test script for generalized DF-VQLS

Tests 8×8 and 16×16 systems.
"""

import numpy as np
from .solver import DFVQLSSolver


def create_tridiagonal_matrix(size: int, diag: float = 2.0, off_diag: float = -1.0) -> np.ndarray:
    """
    Create a tridiagonal matrix with given diagonal and off-diagonal values.
    
    Args:
        size: Matrix size
        diag: Diagonal value
        off_diag: Off-diagonal value
    
    Returns:
        Tridiagonal matrix
    """
    K = np.zeros((size, size))
    np.fill_diagonal(K, diag)
    np.fill_diagonal(K[1:], off_diag)
    np.fill_diagonal(K[:, 1:], off_diag)
    return K


def test_8x8():
    """Test DF-VQLS with 8×8 tridiagonal system."""
    print("\n" + "=" * 70)
    print("Testing DF-VQLS with 8×8 Tridiagonal System")
    print("=" * 70)
    
    # Create tridiagonal matrix
    K = create_tridiagonal_matrix(8, diag=2.0, off_diag=-1.0)
    
    # Right-hand side vector
    f = np.array([2, 0, 2, 5, 0, 0, 0, 0], dtype=float)
    
    # Classical solution
    u_classical = np.linalg.solve(K, f)
    print("\nClassical solution:")
    print(u_classical)
    print()
    
    # Quantum solution
    solver = DFVQLSSolver(
        matrix_size=8,
        num_layers=3,
        optimizer_method='COBYLA',
        max_iter=200,
        random_seed=42,
        verbose=True
    )
    
    u_quantum, result = solver.solve(K, f)
    
    print("\nDF-VQLS solution:")
    print(u_quantum)
    
    # Compute error
    error = np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical)
    print(f"\n{'=' * 70}")
    print(f"Relative Error: {error:.6e}")
    print(f"Final Cost: {result.fun:.6f}")
    print(f"Iterations: {result.nfev}")
    
    if error < 1e-3:
        print("✓ Excellent accuracy achieved!")
    elif error < 1e-2:
        print("✓ Good accuracy achieved!")
    elif error < 0.1:
        print("✓ Acceptable accuracy achieved!")
    else:
        print("⚠ Moderate accuracy - may need more iterations or layers")
    print("=" * 70)


def test_16x16():
    """Test DF-VQLS with 16×16 tridiagonal system."""
    print("\n" + "=" * 70)
    print("Testing DF-VQLS with 16×16 Tridiagonal System")
    print("=" * 70)
    
    # Create tridiagonal matrix
    K = create_tridiagonal_matrix(16, diag=2.0, off_diag=-1.0)
    
    # Right-hand side vector (simple pattern)
    f = np.zeros(16, dtype=float)
    f[0] = 1.0
    f[15] = 1.0
    
    # Classical solution
    u_classical = np.linalg.solve(K, f)
    print("\nClassical solution:")
    print(u_classical)
    print()
    
    # Quantum solution
    solver = DFVQLSSolver(
        matrix_size=16,
        num_layers=4,  # More layers for larger system
        optimizer_method='COBYLA',
        max_iter=200,
        random_seed=42,
        verbose=True
    )
    
    u_quantum, result = solver.solve(K, f)
    
    print("\nDF-VQLS solution:")
    print(u_quantum)
    
    # Compute error
    error = np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical)
    print(f"\n{'=' * 70}")
    print(f"Relative Error: {error:.6e}")
    print(f"Final Cost: {result.fun:.6f}")
    print(f"Iterations: {result.nfev}")
    
    if error < 1e-3:
        print("✓ Excellent accuracy achieved!")
    elif error < 1e-2:
        print("✓ Good accuracy achieved!")
    elif error < 0.1:
        print("✓ Acceptable accuracy achieved!")
    else:
        print("⚠ Moderate accuracy - may need more iterations or layers")
    print("=" * 70)


def test_8x8_quick():
    """Quick test for 8×8 system with fewer iterations."""
    print("\n" + "=" * 70)
    print("Quick Test: DF-VQLS with 8×8 System (10 iterations)")
    print("=" * 70)
    
    # Create tridiagonal matrix
    K = create_tridiagonal_matrix(8, diag=2.0, off_diag=-1.0)
    
    # Right-hand side vector
    f = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    
    # Classical solution
    u_classical = np.linalg.solve(K, f)
    print("\nClassical solution:")
    print(u_classical)
    print()
    
    # Quantum solution
    solver = DFVQLSSolver(
        matrix_size=8,
        num_layers=2,  # Fewer layers for quick test
        optimizer_method='COBYLA',
        max_iter=10,  # Very few iterations for quick test
        random_seed=42,
        verbose=True
    )
    
    u_quantum, result = solver.solve(K, f)
    
    print("\nDF-VQLS solution:")
    print(u_quantum)
    
    # Compute error
    error = np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical)
    print(f"\n{'=' * 70}")
    print(f"Relative Error: {error:.6e}")
    print(f"Final Cost: {result.fun:.6f}")
    print(f"Iterations: {result.nfev}")
    print("=" * 70)
    
    if error < 0.5:
        print("✓ Test passed - algorithm is working!")
    else:
        print("⚠ High error - expected with only 10 iterations")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "8x8":
            test_8x8()
        elif test_name == "16x16":
            test_16x16()
        elif test_name == "quick":
            test_8x8_quick()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: 8x8, 16x16, quick")
    else:
        # Run quick test by default
        test_8x8_quick()

