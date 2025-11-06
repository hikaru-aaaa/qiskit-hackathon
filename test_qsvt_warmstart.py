"""
Test QSVT → DF-VQLS Warm Start Integration

Validates the complete pipeline:
1. QSVT computes initial solution
2. Convert QSVT solution to ansatz parameters
3. DF-VQLS uses warm start parameters
4. Compare iterations: random init vs warm start
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add qsvt to path for relative imports
qsvt_dir = Path(__file__).parent / "src" / "qsvt"
sys.path.insert(0, str(qsvt_dir))

from src.linear_solvers import DFVQLSSolver, QSVTSolver
from src.vqls.generalized.utils import qsvt_to_theta_initialization
from src.vqls.generalized.ansatz import HardwareEfficientAnsatz

print("=" * 70)
print("QSVT → DF-VQLS WARM START TEST")
print("=" * 70)

# Define 2×2 problem
A = np.array([[3, 1], [1, 3]], dtype=float)
b = np.array([1, 2], dtype=float)
x_classical = np.linalg.solve(A, b)

print(f"\nProblem: Ax = b")
print(f"A = \n{A}")
print(f"b = {b}")
print(f"Classical solution: {x_classical}")

# Step 1: Run QSVT
print("\n" + "-" * 70)
print("Step 1: QSVT Solver")
print("-" * 70)

qsvt_solver = QSVTSolver(matrix_size=2, verbose=False)
t0 = time.time()
x_qsvt, qsvt_meta = qsvt_solver.solve(A, b)
t_qsvt = time.time() - t0

qsvt_error = np.linalg.norm(x_qsvt - x_classical) / np.linalg.norm(x_classical)
print(f"QSVT solution: {x_qsvt}")
print(f"QSVT error: {qsvt_error:.6e}")
print(f"QSVT time: {t_qsvt:.2f}s")

# Step 2: Convert QSVT solution to parameters
print("\n" + "-" * 70)
print("Step 2: Convert QSVT → Ansatz Parameters")
print("-" * 70)

ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=2)
t0 = time.time()
theta_init = qsvt_to_theta_initialization(
    x_qsvt=x_qsvt,
    ansatz=ansatz,
    num_qubits=1,
    max_iter=50,
    verbose=True
)
t_convert = time.time() - t0

print(f"Conversion time: {t_convert:.2f}s")
print(f"Parameters shape: {theta_init.shape}")

# Step 3a: DF-VQLS with random init (baseline)
print("\n" + "-" * 70)
print("Step 3a: DF-VQLS with Random Init (Baseline)")
print("-" * 70)

solver_random = DFVQLSSolver(
    matrix_size=2,
    num_layers=2,
    max_iter=100,
    random_seed=42,
    verbose=False
)

t0 = time.time()
x_random, result_random = solver_random.solve(A, b)
t_random = time.time() - t0

error_random = np.linalg.norm(x_random - x_classical) / np.linalg.norm(x_classical)
print(f"Solution: {x_random}")
print(f"Error: {error_random:.6e}")
print(f"Iterations: {result_random['iterations']}")
print(f"Time: {t_random:.2f}s")
iter_random = result_random['iterations']

# Step 3b: DF-VQLS with QSVT warm start
print("\n" + "-" * 70)
print("Step 3b: DF-VQLS with QSVT Warm Start")
print("-" * 70)

solver_warmstart = DFVQLSSolver(
    matrix_size=2,
    num_layers=2,
    max_iter=100,
    random_seed=42,
    verbose=False
)

t0 = time.time()
x_warmstart, result_warmstart = solver_warmstart.solve(A, b, initial_params=theta_init)
t_warmstart = time.time() - t0

error_warmstart = np.linalg.norm(x_warmstart - x_classical) / np.linalg.norm(x_classical)
print(f"Solution: {x_warmstart}")
print(f"Error: {error_warmstart:.6e}")
print(f"Iterations: {result_warmstart['iterations']}")
print(f"Time: {t_warmstart:.2f}s")
iter_warmstart = result_warmstart['iterations']

# Step 4: Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print(f"\n{'Method':<20} {'Iterations':<15} {'Error':<15} {'Time (s)':<10}")
print("-" * 70)
print(f"{'Classical':<20} {'N/A':<15} {0.0:<15.6e} {'instant':<10}")
print(f"{'QSVT':<20} {'N/A':<15} {qsvt_error:<15.6e} {t_qsvt:<10.2f}")
print(f"{'DF-VQLS (random)':<20} {iter_random:<15} {error_random:<15.6e} {t_random:<10.2f}")
print(f"{'DF-VQLS (warm)':<20} {iter_warmstart:<15} {error_warmstart:<15.6e} {t_warmstart:<10.2f}")

iteration_reduction = (iter_random - iter_warmstart) / iter_random * 100
time_saved = t_random - t_warmstart

print(f"\n{'Metric':<40} {'Value':<20}")
print("-" * 70)
print(f"{'Iteration reduction':<40} {iteration_reduction:.1f}%")
print(f"{'DF-VQLS time saved':<40} {time_saved:.2f}s")
print(f"{'Conversion overhead':<40} {t_convert:.2f}s")
print(f"{'Net benefit':<40} {time_saved - t_convert:.2f}s")

# Total time comparison
total_random = t_random
total_hybrid = t_qsvt + t_convert + t_warmstart
print(f"\n{'Total Time Comparison':<40}")
print("-" * 70)
print(f"{'Pure DF-VQLS (random init)':<40} {total_random:.2f}s")
print(f"{'Hybrid (QSVT + warm start)':<40} {total_hybrid:.2f}s")
print(f"{'Difference':<40} {total_hybrid - total_random:.2f}s")

# Success criteria
print("\n" + "=" * 70)
print("SUCCESS CRITERIA")
print("=" * 70)

success = True
if iteration_reduction >= 10:
    print(f"✅ Iteration reduction: {iteration_reduction:.1f}% (target: ≥10%)")
else:
    print(f"⚠️  Iteration reduction: {iteration_reduction:.1f}% (target: ≥10%)")
    success = False

if error_warmstart <= error_random * 1.1:  # Within 10% of random
    print(f"✅ Solution quality maintained: {error_warmstart:.6e}")
else:
    print(f"⚠️  Solution quality degraded: {error_warmstart:.6e}")
    success = False

if iter_warmstart < iter_random:
    print(f"✅ Fewer iterations: {iter_warmstart} < {iter_random}")
else:
    print(f"⚠️  Same or more iterations: {iter_warmstart} vs {iter_random}")
    success = False

print("\n" + "=" * 70)
if success:
    print("✅ WARM START FEATURE VALIDATED SUCCESSFULLY!")
else:
    print("⚠️  Warm start shows benefit but below target")
print("=" * 70)
