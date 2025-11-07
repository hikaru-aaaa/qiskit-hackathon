"""
Test BFGS vs COBYLA on the problematic 8x8 system.

This is the key test to verify that gradient-based optimization (BFGS)
solves the convergence failure that occurs with COBYLA on 8x8 problems.
"""

import numpy as np
from src.vqls.generalized.solver import DFVQLSSolver
from test_combined import create_problem_8x8

def test_8x8_comparison():
    """Compare BFGS vs COBYLA on 8x8 problem."""
    print("="*80)
    print("8x8 Problem: BFGS vs COBYLA Comparison")
    print("="*80)

    # Create 8x8 problem
    A, b, desc = create_problem_8x8()
    print(f"\nProblem: {desc}")
    print(f"Matrix shape: {A.shape}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")

    # Compute reference solution
    x_true = np.linalg.solve(A, b)
    print(f"Reference solution norm: {np.linalg.norm(x_true):.6f}")

    # Test parameters
    num_layers = 3
    max_iter = 50  # Reduce for faster testing

    print("\n" + "="*80)
    print("TEST 1: COBYLA (Gradient-Free)")
    print("="*80)

    # Create COBYLA solver
    solver_cobyla = DFVQLSSolver(
        matrix_size=8,
        num_layers=num_layers,
        optimizer_method='COBYLA',
        max_iter=max_iter,
        verbose=True
    )

    # Solve with COBYLA
    x_cobyla, result_cobyla, _ = solver_cobyla.solve(A, b, track_iterations=True)

    # Compute error
    error_cobyla = np.linalg.norm(x_cobyla - x_true) / np.linalg.norm(x_true)
    print(f"\nCOBYLA Results:")
    print(f"  Final cost: {result_cobyla.fun:.6f}")
    print(f"  Relative error: {error_cobyla:.6f}")
    print(f"  Number of iterations: {result_cobyla.nfev}")
    print(f"  Success: {result_cobyla.success}")

    print("\n" + "="*80)
    print("TEST 2: BFGS (Gradient-Based)")
    print("="*80)

    # Create BFGS solver
    solver_bfgs = DFVQLSSolver(
        matrix_size=8,
        num_layers=num_layers,
        optimizer_method='BFGS',
        max_iter=max_iter,
        verbose=True
    )

    # Solve with BFGS
    x_bfgs, result_bfgs, _ = solver_bfgs.solve(A, b, track_iterations=True)

    # Compute error
    error_bfgs = np.linalg.norm(x_bfgs - x_true) / np.linalg.norm(x_true)
    print(f"\nBFGS Results:")
    print(f"  Final cost: {result_bfgs.fun:.6f}")
    print(f"  Relative error: {error_bfgs:.6f}")
    print(f"  Number of iterations: {result_bfgs.nfev}")
    print(f"  Success: {result_bfgs.success}")

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<30} {'COBYLA':<20} {'BFGS':<20}")
    print("-"*80)
    print(f"{'Final Cost':<30} {result_cobyla.fun:<20.6f} {result_bfgs.fun:<20.6f}")
    print(f"{'Relative Error':<30} {error_cobyla:<20.6f} {error_bfgs:<20.6f}")
    print(f"{'Iterations':<30} {result_cobyla.nfev:<20} {result_bfgs.nfev:<20}")
    print(f"{'Success':<30} {str(result_cobyla.success):<20} {str(result_bfgs.success):<20}")

    # Determine improvement
    improvement = error_cobyla / error_bfgs if error_bfgs > 0 else float('inf')
    print(f"\nBFGS Improvement: {improvement:.2f}x better error than COBYLA")

    # Success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)

    success = True

    # Check BFGS error
    if error_bfgs < 0.1:
        print("✓ BFGS error < 0.1 (PASS)")
    else:
        print(f"✗ BFGS error = {error_bfgs:.6f} >= 0.1 (FAIL)")
        success = False

    # Check BFGS better than COBYLA
    if error_bfgs < error_cobyla:
        print("✓ BFGS error < COBYLA error (PASS)")
    else:
        print(f"✗ BFGS error >= COBYLA error (FAIL)")
        success = False

    # Check BFGS convergence
    if result_bfgs.fun < 0.5:  # Cost should be low
        print(f"✓ BFGS cost = {result_bfgs.fun:.6f} < 0.5 (PASS)")
    else:
        print(f"✗ BFGS cost = {result_bfgs.fun:.6f} >= 0.5 (FAIL)")
        success = False

    return success, error_cobyla, error_bfgs


if __name__ == "__main__":
    success, error_cobyla, error_bfgs = test_8x8_comparison()

    print("\n" + "="*80)
    if success:
        print("8x8 TEST PASSED ✓")
        print(f"BFGS successfully improved convergence: {error_bfgs:.6f} vs {error_cobyla:.6f}")
    else:
        print("8x8 TEST FAILED ✗")
        print("BFGS did not achieve expected improvement")
    print("="*80)
