"""
Test script for generalized DF-VQLS

Tests 8×8 and 16×16 systems.
"""

import numpy as np
import time
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


def test_parallel_performance():
    """Compare performance between sequential and parallel execution."""
    print("\n" + "=" * 70)
    print("Performance Comparison: Sequential vs Parallel Execution")
    print("=" * 70)
    
    # Create test data
    K = create_tridiagonal_matrix(8, diag=2.0, off_diag=-1.0)
    f = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    
    # Test parameters
    max_iter = 5  # Small number for quick comparison
    num_layers = 2
    
    print(f"\nTest configuration:")
    print(f"  Matrix size: 8×8")
    print(f"  Iterations: {max_iter}")
    print(f"  Ansatz layers: {num_layers}")
    print()
    
    # Sequential execution
    print("-" * 70)
    print("Sequential Execution (use_parallel=False)")
    print("-" * 70)
    solver_seq = DFVQLSSolver(
        matrix_size=8,
        num_layers=num_layers,
        optimizer_method='COBYLA',
        max_iter=max_iter,
        random_seed=42,
        verbose=False,  # Reduce output for cleaner comparison
        use_parallel=False
    )
    
    start_seq = time.time()
    u_seq, result_seq = solver_seq.solve(K, f)
    time_seq = time.time() - start_seq
    
    print(f"Time: {time_seq:.2f} seconds")
    print(f"Final cost: {result_seq.fun:.6f}")
    print(f"Iterations: {result_seq.nfev}")
    
    # Parallel execution
    print("\n" + "-" * 70)
    print("Parallel Execution (use_parallel=True)")
    print("-" * 70)
    solver_par = DFVQLSSolver(
        matrix_size=8,
        num_layers=num_layers,
        optimizer_method='COBYLA',
        max_iter=max_iter,
        random_seed=42,
        verbose=False,  # Reduce output for cleaner comparison
        use_parallel=True
    )
    
    start_par = time.time()
    u_par, result_par = solver_par.solve(K, f)
    time_par = time.time() - start_par
    
    print(f"Time: {time_par:.2f} seconds")
    print(f"Final cost: {result_par.fun:.6f}")
    print(f"Iterations: {result_par.nfev}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"Sequential time: {time_seq:.2f} seconds")
    print(f"Parallel time:   {time_par:.2f} seconds")
    
    if time_par < time_seq:
        speedup = time_seq / time_par
        improvement = (1 - time_par / time_seq) * 100
        print(f"✓ Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")
    else:
        slowdown = time_par / time_seq
        overhead = (time_par / time_seq - 1) * 100
        print(f"⚠ Slowdown: {slowdown:.2f}x ({overhead:.1f}% slower)")
        print("  (Parallel overhead may be due to GIL or initialization cost)")
    
    # Verify solutions are similar
    error_diff = np.linalg.norm(u_seq - u_par) / np.linalg.norm(u_seq)
    print(f"\nSolution difference: {error_diff:.2e}")
    if error_diff < 1e-6:
        print("✓ Solutions match (within numerical precision)")
    else:
        print("⚠ Solutions differ (may be due to different random seeds or optimization path)")
    
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
        elif test_name == "parallel":
            test_parallel_performance()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: 8x8, 16x16, quick, parallel")
    else:
        # Run quick test by default
        test_8x8_quick()

