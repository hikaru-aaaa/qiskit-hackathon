"""
QSVTとDF-VQLSのノイズ耐性比較

Dephasingノイズ（位相減衰）を追加して、ノイズ強度に対する耐性を比較します。
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, PhaseDampingError

# Add src/qsvt to Python path for inverse_matrix import
sys.path.insert(0, str(Path(__file__).parent / "src" / "qsvt"))

from src.linear_solvers import DFVQLSSolver
from src.qsvt.lse_solver import LSESolver
from src.utils.preconditioning import (
    jacobi_precondition,
    recover_solution,
    print_condition_number_analysis,
)
from test_combined import (
    create_matrix_with_condition_number,
)


@dataclass
class NoiseResult:
    noise_strength: float
    error: float


@dataclass
class NoiseResultList:
    name: str
    results: list[NoiseResult]


def calculate_qc_depth(qc: QuantumCircuit) -> int:
    depth = 0
    while qc.depth() > depth:
        qc = qc.decompose()
        depth = qc.depth()
    return depth


def create_noise_model(phase_damping_rate: float) -> NoiseModel:
    """
    Create a noise model with phase damping (dephasing) noise.
    
    Args:
        phase_damping_rate: Phase damping rate (0 ≤ γ ≤ 1)
    
    Returns:
        NoiseModel with phase damping error
    """
    noise_model = NoiseModel()
    phase_damping_error = PhaseDampingError(phase_damping_rate)
    noise_model.add_all_qubit_quantum_error(phase_damping_error, ['u', 'rx', 'ry', 'rz', 'h', 'cx', 'cz'])
    return noise_model


def collect_qsvt_results_with_noise(
    A: np.ndarray,
    b: np.ndarray,
    target_depth: int,
    noise_strengths: list[float]
) -> NoiseResultList:
    """
    Collect QSVT results with different noise strengths at fixed depth.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        target_depth: Target depth to match
        noise_strengths: List of noise strengths (phase damping rates)
    
    Returns:
        NoiseResultList with results for each noise strength
    """
    results = NoiseResultList(name="QSVT", results=[])
    
    # First, find kappa that gives approximately target_depth
    print(f"QSVT: Finding kappa for target depth {target_depth}...")
    best_kappa = None
    best_depth_diff = float('inf')
    
    for kappa in range(1, 20):
        qsvt_solver = LSESolver(A, b, kappa=kappa)
        x_solution, qsvt_circuit = qsvt_solver.solve_linear_system_quantum()
        depth = calculate_qc_depth(qsvt_circuit)
        depth_diff = abs(depth - target_depth)
        
        if depth_diff < best_depth_diff:
            best_depth_diff = depth_diff
            best_kappa = kappa
            if depth >= target_depth:
                break
    
    actual_depth = calculate_qc_depth(LSESolver(A, b, kappa=best_kappa).solve_linear_system_quantum()[1])
    print(f"QSVT: Using kappa={best_kappa} (depth={actual_depth})")
    
    # Get classical solution for error calculation
    classical_solution = np.linalg.solve(A, b)
    
    # Run with different noise strengths
    for noise_strength in noise_strengths:
        print(f"QSVT: Running with noise strength {noise_strength:.3f}...")
        
        # Create noise model
        noise_model = create_noise_model(noise_strength) if noise_strength > 0 else None
        
        # Solve with QSVT (with noise model if provided)
        qsvt_solver = LSESolver(A, b, kappa=best_kappa, noise_model=noise_model)
        x_solution, qsvt_circuit = qsvt_solver.solve_linear_system_quantum(statevector=True)
        
        # Calculate error
        error = np.linalg.norm(x_solution - classical_solution) / np.linalg.norm(
            classical_solution
        )
        
        results.results.append(NoiseResult(noise_strength, error))
        print(f"  Noise strength {noise_strength:.3f}: error = {error:.6e}")
    
    return results


def collect_dfvqls_results_with_noise(
    A: np.ndarray,
    b: np.ndarray,
    target_depth: int,
    noise_strengths: list[float],
    use_preconditioning: bool = False,
    optimizer_method: str = "BFGS"
) -> NoiseResultList:
    """
    Collect DF-VQLS results with different noise strengths at fixed depth.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        target_depth: Target depth to match
        noise_strengths: List of noise strengths (phase damping rates)
        use_preconditioning: If True, apply Jacobi preconditioning
        optimizer_method: Optimization method
    
    Returns:
        NoiseResultList with results for each noise strength
    """
    results = NoiseResultList(
        name="DF-VQLS" + (" (Precond)" if use_preconditioning else ""),
        results=[]
    )
    
    # Apply preconditioning if requested
    if use_preconditioning:
        A_work, b_work, D_sqrt_inv = jacobi_precondition(A, b)
        print_condition_number_analysis(A, A_work, "K")
    else:
        A_work, b_work, D_sqrt_inv = A, b, None
    
    # Determine per-iteration depth (using noiseless simulation)
    print("DF-VQLS: Calculating per-iteration circuit depth...")
    temp_solver = DFVQLSSolver(
        matrix_size=len(A),
        num_layers=3,
        optimizer_method=optimizer_method,
        max_iter=1,
        verbose=False,
    )
    _, _, (num_circuit, den_circuit) = temp_solver.solve(A_work, b_work)
    per_iter_depth = max(calculate_qc_depth(num_circuit), calculate_qc_depth(den_circuit))
    print(f"  Per-iteration depth: {per_iter_depth}")
    
    # Calculate number of iterations to reach target depth
    num_iterations = target_depth // per_iter_depth
    print(f"  Running {num_iterations} iterations to reach depth ~{num_iterations * per_iter_depth}")
    
    # Get classical solution for error calculation
    classical_solution = np.linalg.solve(A, b)
    
    # Run with different noise strengths
    for noise_strength in noise_strengths:
        print(f"DF-VQLS: Running with noise strength {noise_strength:.3f}...")
        
        # Create noise model (None if noise_strength is 0)
        noise_model = create_noise_model(noise_strength) if noise_strength > 0 else None
        
        # Create solver with noise model
        dfvqls_solver = DFVQLSSolver(
            matrix_size=len(A),
            num_layers=3,
            optimizer_method=optimizer_method,
            max_iter=num_iterations,
            verbose=False,
            noise_model=noise_model,  # Pass noise model to solver
        )
        
        # Run optimization with random initialization
        num_params = dfvqls_solver.solver.ansatz.num_parameters()
        np.random.seed(42)  # Fixed seed for reproducibility
        initial_params = np.random.uniform(0, 2 * np.pi, num_params)
        
        x_solution, metadata, (num_circuit, den_circuit) = dfvqls_solver.solve(
            A_work, b_work, initial_params=initial_params, track_iterations=True
        )
        
        # Recover original solution if preconditioning was used
        if use_preconditioning:
            x_solution = recover_solution(x_solution, D_sqrt_inv)
        
        # Calculate error
        error = np.linalg.norm(x_solution - classical_solution) / np.linalg.norm(
            classical_solution
        )
        
        results.results.append(NoiseResult(noise_strength, error))
        print(f"  Noise strength {noise_strength:.3f}: error = {error:.6e}")
    
    return results


def save_noise_result_list(result_list: NoiseResultList, filename: str) -> None:
    """Save NoiseResultList to JSON file."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    data = asdict(result_list)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Result saved to {filename}")


def draw_noise_plot(
    result_list1: NoiseResultList,
    result_list2: NoiseResultList,
    title: str,
    output_dir: str = "output/noise"
) -> None:
    """Draw comparison plot for noise tolerance."""
    plt.figure(figsize=(10, 6))
    
    # Extract data
    noise_strengths1 = [r.noise_strength for r in result_list1.results]
    errors1 = [r.error for r in result_list1.results]
    
    noise_strengths2 = [r.noise_strength for r in result_list2.results]
    errors2 = [r.error for r in result_list2.results]
    
    # Plot
    plt.plot(
        noise_strengths1,
        errors1,
        marker='o',
        linestyle='-',
        label=result_list1.name,
        linewidth=2,
        markersize=8
    )
    
    plt.plot(
        noise_strengths2,
        errors2,
        marker='s',
        linestyle='-',
        label=result_list2.name,
        linewidth=2,
        markersize=8
    )
    
    plt.xlabel("Noise Strength (Phase Damping Rate)", fontsize=14, fontweight='bold')
    plt.ylabel("Relative Error", fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')  # Log scale for error
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/{title}.pdf", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_dir}/{title}.pdf")
    plt.close()


def main() -> None:
    # Parameters
    matrix_size = 4
    kappa = 3
    target_depth = 1000
    noise_strengths = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    
    print("=" * 80)
    print(f"Noise Tolerance Comparison: {matrix_size}×{matrix_size}, κ={kappa}")
    print(f"Target depth: {target_depth}")
    print(f"Noise strengths: {noise_strengths}")
    print("=" * 80)
    
    # Create problem
    A, b = create_matrix_with_condition_number(matrix_size, kappa)
    title = f"{matrix_size}x{matrix_size}_kappa={kappa}_depth={target_depth}"
    
    # Collect QSVT results
    print("\n" + "=" * 80)
    print("Collecting QSVT results with noise...")
    print("=" * 80)
    qsvt_results = collect_qsvt_results_with_noise(
        A, b, target_depth, noise_strengths
    )
    save_noise_result_list(qsvt_results, f"output/noise/qsvt_{title}.json")
    
    # Collect DF-VQLS results
    print("\n" + "=" * 80)
    print("Collecting DF-VQLS results with noise...")
    print("=" * 80)
    dfvqls_results = collect_dfvqls_results_with_noise(
        A, b, target_depth, noise_strengths,
        use_preconditioning=True,
        optimizer_method="BFGS"
    )
    save_noise_result_list(dfvqls_results, f"output/noise/dfvqls_{title}.json")
    
    # Draw comparison plot
    print("\n" + "=" * 80)
    print("Drawing comparison plot...")
    print("=" * 80)
    draw_noise_plot(qsvt_results, dfvqls_results, title)
    
    print("\n" + "=" * 80)
    print("Noise tolerance comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

