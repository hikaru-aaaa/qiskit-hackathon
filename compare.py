import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

# Add src/qsvt to Python path for inverse_matrix import
sys.path.insert(0, str(Path(__file__).parent / "src" / "qsvt"))

from src.linear_solvers import DFVQLSSolver
from src.qsvt.lse_solver import LSESolver
from test_combined import (
    create_matrix_with_condition_number,
)


@dataclass
class Result:
    depth: int
    error: float


@dataclass
class ResultList:
    name: str
    results: list[Result]


def calculate_qc_depth(qc: QuantumCircuit) -> int:
    depth = 0
    while qc.depth() > depth:
        qc = qc.decompose()
        depth = qc.depth()
    return depth


def collect_qsvt_results(A: np.ndarray, b: np.ndarray, title: str, use_depth_matched_dir: bool = False) -> ResultList:
    kappa_list = [1, 2, 3, 4, 5, 6, 7, 8]
    results = ResultList(name="QSVT", results=[])
    for kappa in kappa_list:
        qsvt_solver = LSESolver(A, b, kappa=kappa)
        x_solution, qsvt_circuit = qsvt_solver.solve_linear_system_quantum()
        depth = calculate_qc_depth(qsvt_circuit)
        classical_solution = np.linalg.solve(A, b)
        error = np.linalg.norm(x_solution - classical_solution) / np.linalg.norm(
            classical_solution
        )
        results.results.append(Result(depth, error))

    output_dir = "output/depth_matched" if use_depth_matched_dir else "output"
    save_result_list(results, f"{output_dir}/qsvt_{title}.json")
    return results


def collect_dfvqls_results(A: np.ndarray, b: np.ndarray, title: str, qsvt_depth: int = None) -> ResultList:
    """
    Collect DF-VQLS results, optionally matching QSVT depth budget.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        title: Title for saving results
        qsvt_depth: If provided, run DF-VQLS until total depth matches this value
    """
    results = ResultList(name="DF-VQLS", results=[])

    # First, determine DF-VQLS per-iteration depth with a quick run
    print("DF-VQLS: Calculating per-iteration circuit depth...")
    temp_solver = DFVQLSSolver(
        matrix_size=len(A),
        num_layers=2,
        optimizer_method="COBYLA",
        max_iter=1,
        verbose=False,
    )
    _, _, (num_circuit, den_circuit) = temp_solver.solve(A, b)
    depth_num = calculate_qc_depth(num_circuit)
    depth_den = calculate_qc_depth(den_circuit)
    per_iter_depth = max(depth_num, depth_den)
    print(f"  Per-iteration depth: {per_iter_depth}")

    # Determine max_iter based on depth matching or default checkpoints
    if qsvt_depth is not None:
        # Match QSVT depth budget
        max_iter = qsvt_depth // per_iter_depth
        print(f"  QSVT depth budget: {qsvt_depth}")
        print(f"  Running DF-VQLS for {max_iter} iterations to match QSVT depth")
        # Create evenly distributed checkpoints
        checkpoint_iters = [
            max_iter // 10,
            max_iter // 5,
            max_iter // 2,
            max_iter
        ]
        checkpoint_iters = [c for c in checkpoint_iters if c > 0]  # Remove any zeros
    else:
        # Use default checkpoints
        checkpoint_iters = [100, 200, 300, 400, 500]
        max_iter = max(checkpoint_iters)
        print(f"DF-VQLS: Running with default checkpoints up to {max_iter} iterations")

    # Run main optimization with iteration tracking enabled
    dfvqls_solver = DFVQLSSolver(
        matrix_size=len(A),
        num_layers=2,
        optimizer_method="COBYLA",
        max_iter=max_iter,
        verbose=False,
    )

    print(f"\nDF-VQLS: Running optimization with max_iter={max_iter}")
    x_solution, metadata, (num_circuit, den_circuit) = dfvqls_solver.solve(
        A, b, track_iterations=True
    )

    depth = per_iter_depth  # Use pre-calculated depth

    # Get classical solution for error calculation
    classical_solution = np.linalg.solve(A, b)

    # Extract iteration history
    iteration_history = metadata.get("iteration_history", [])

    # Check if COBYLA converged early
    actual_iterations = len(iteration_history)
    if actual_iterations < max_iter:
        print(
            f"Note: COBYLA converged early at iteration {actual_iterations}/{max_iter}"
        )

    # Extract results at each checkpoint
    for checkpoint in checkpoint_iters:
        # Find the iteration closest to the checkpoint (0-indexed)
        checkpoint_idx = checkpoint - 1

        # Handle early convergence: use last available iteration if checkpoint not reached
        if checkpoint_idx >= len(iteration_history):
            # COBYLA converged before this checkpoint - use final converged parameters
            checkpoint_data = iteration_history[-1]
            actual_iter = checkpoint_data["iteration"] + 1
            print(
                f"      Checkpoint {checkpoint}: using converged solution from iter {actual_iter}"
            )
        else:
            checkpoint_data = iteration_history[checkpoint_idx]
            actual_iter = checkpoint_data["iteration"] + 1

        # Reconstruct solution from parameters at this checkpoint
        checkpoint_params = checkpoint_data["params"]
        x_checkpoint = dfvqls_solver.get_solution_at_params(checkpoint_params, A, b)

        # Calculate error at this checkpoint
        error = np.linalg.norm(x_checkpoint - classical_solution) / np.linalg.norm(
            classical_solution
        )

        # Store result (depth × actual_iteration for fair comparison)
        results.results.append(Result(depth * actual_iter, error))
        print(
            f"DF-VQLS: iter={actual_iter}, depth={depth * actual_iter}, "
            f"error={error:.4f}, cost={checkpoint_data['cost']:.6f}"
        )

    output_dir = "output/depth_matched" if qsvt_depth is not None else "output"
    save_result_list(results, f"{output_dir}/dfvqls_{title}.json")
    print(f"\nTotal optimization completed in {actual_iterations} iterations")
    return results


def save_result_list(result_list: ResultList, filename: str) -> None:
    """Save a single ResultList to JSON file."""
    # Ensure the output directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    data = asdict(result_list)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Result saved to {filename}")


def load_result_list(filename: str) -> ResultList:
    """Load a single ResultList from JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    result_list = ResultList(
        name=data["name"],
        results=[Result(**r) for r in data["results"]],
    )
    return result_list


def draw_result_plot(
    result_list1: ResultList, result_list2: ResultList, title: str, use_depth_matched_dir: bool = False
) -> None:
    plt.plot(
        [result.depth for result in result_list1.results],
        [result.error for result in result_list1.results],
        label=result_list1.name,
    )
    plt.plot(
        [result.depth for result in result_list2.results],
        [result.error for result in result_list2.results],
        label=result_list2.name,
    )
    plt.legend()
    plt.title(title)
    plt.xlabel("Depth")
    plt.ylabel("Error")
    output_dir = "output/depth_matched" if use_depth_matched_dir else "output"
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/{title}.pdf")
    plt.close()


def main() -> None:
    kappa = 4

    # 2x2 system
    print("=" * 80)
    print("Testing 2x2 system")
    print("=" * 80)
    title = f"2x2_kappa={kappa}"
    A, b = create_matrix_with_condition_number(2, kappa)
    qsvt_results = collect_qsvt_results(A, b, title, use_depth_matched_dir=True)
    # Get max depth from QSVT results
    qsvt_max_depth = max(result.depth for result in qsvt_results.results)
    print(f"\nQSVT maximum depth: {qsvt_max_depth}")
    print(f"Running DF-VQLS with matched depth budget...\n")
    dfvqls_results = collect_dfvqls_results(A, b, title, qsvt_depth=qsvt_max_depth)
    draw_result_plot(qsvt_results, dfvqls_results, title, use_depth_matched_dir=True)

    # 4x4 system
    print("\n" + "=" * 80)
    print("Testing 4x4 system")
    print("=" * 80)
    title = f"4x4_kappa={kappa}"
    A, b = create_matrix_with_condition_number(4, kappa)
    qsvt_results = collect_qsvt_results(A, b, title, use_depth_matched_dir=True)
    qsvt_max_depth = max(result.depth for result in qsvt_results.results)
    print(f"\nQSVT maximum depth: {qsvt_max_depth}")
    print(f"Running DF-VQLS with matched depth budget...\n")
    dfvqls_results = collect_dfvqls_results(A, b, title, qsvt_depth=qsvt_max_depth)
    draw_result_plot(qsvt_results, dfvqls_results, title, use_depth_matched_dir=True)

    # 8x8 system
    print("\n" + "=" * 80)
    print("Testing 8x8 system")
    print("=" * 80)
    title = f"8x8_kappa={kappa}"
    A, b = create_matrix_with_condition_number(8, kappa)
    qsvt_results = collect_qsvt_results(A, b, title, use_depth_matched_dir=True)
    qsvt_max_depth = max(result.depth for result in qsvt_results.results)
    print(f"\nQSVT maximum depth: {qsvt_max_depth}")
    print(f"Running DF-VQLS with matched depth budget...\n")
    dfvqls_results = collect_dfvqls_results(A, b, title, qsvt_depth=qsvt_max_depth)
    draw_result_plot(qsvt_results, dfvqls_results, title, use_depth_matched_dir=True)

    print("\n" + "=" * 80)
    print("All comparisons complete!")
    print("=" * 80)
    return


if __name__ == "__main__":
    # kappa = 4
    # A, b = create_matrix_with_condition_number(2, kappa)
    # u, s, vh = np.linalg.svd(A)
    # kappa = s.max() / s.min()
    # print("2x2: ", kappa)
    # A, b = create_matrix_with_condition_number(4, kappa)
    # u, s, vh = np.linalg.svd(A)
    # kappa = s.max() / s.min()
    # print("4x4: ", kappa)
    # A, b = create_matrix_with_condition_number(8, kappa)
    # u, s, vh = np.linalg.svd(A)
    # kappa = s.max() / s.min()
    # print("8x8: ", kappa)
    main()
