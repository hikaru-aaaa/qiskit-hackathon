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


def collect_qsvt_results(A: np.ndarray, b: np.ndarray, title: str) -> ResultList:
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

    save_result_list(results, f"output/qsvt_{title}.json")
    return results


def collect_dfvqls_results(A: np.ndarray, b: np.ndarray, title: str) -> ResultList:
    checkpoint_iters = [100, 200, 300, 400, 500]
    max_iter = max(checkpoint_iters)
    results = ResultList(name="DF-VQLS", results=[])

    # Single optimization run with iteration tracking enabled
    dfvqls_solver = DFVQLSSolver(
        matrix_size=len(A),
        num_layers=2,
        optimizer_method="COBYLA",
        max_iter=max_iter,
        verbose=False,
    )

    print(f"DF-VQLS: Running single optimization with max_iter={max_iter}")

    x_solution, metadata, (num_circuit, den_circuit) = dfvqls_solver.solve(
        A, b, track_iterations=True
    )

    depth_num = calculate_qc_depth(num_circuit)
    depth_den = calculate_qc_depth(den_circuit)
    depth = max(depth_num, depth_den)

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

    save_result_list(results, f"output/dfvqls_{title}.json")
    print(f"\nTotal optimization completed in {actual_iterations} iterations")
    return results


def save_result_list(result_list: ResultList, filename: str) -> None:
    """Save a single ResultList to JSON file."""
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
    result_list1: ResultList, result_list2: ResultList, title: str
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
    plt.savefig(f"output/{title}.pdf")
    plt.close()


def main() -> None:
    kappa = 4
    title = f"2x2_kappa={kappa}"
    A, b = create_matrix_with_condition_number(2, kappa)
    qsvt_results = collect_qsvt_results(A, b, title)
    dfvqls_results = collect_dfvqls_results(A, b, title)
    draw_result_plot(qsvt_results, dfvqls_results, title)

    title = f"4x4_kappa={kappa}"
    A, b = create_matrix_with_condition_number(4, kappa)
    qsvt_results = collect_qsvt_results(A, b, title)
    dfvqls_results = collect_dfvqls_results(A, b, title)
    draw_result_plot(qsvt_results, dfvqls_results, title)

    title = f"8x8_kappa={kappa}"
    A, b = create_matrix_with_condition_number(8, kappa)
    qsvt_results = collect_qsvt_results(A, b, title)
    dfvqls_results = collect_dfvqls_results(A, b, title)
    draw_result_plot(qsvt_results, dfvqls_results, title)

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
