import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

# Add src/qsvt to Python path for inverse_matrix import
sys.path.insert(0, str(Path(__file__).parent / "src" / "qsvt"))

from src.linear_solvers import DFVQLSSolver
from src.qsvt.lse_solver import LSESolver
from src.utils.preconditioning import (
    jacobi_precondition,
    print_condition_number_analysis,
    recover_solution,
)
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


def collect_qsvt_results(
    A: np.ndarray,
    b: np.ndarray,
    title: str,
    simulator: AerSimulator,
    shot: int,
    use_depth_matched_dir: bool = False,
) -> ResultList:
    kappa_list = [1, 2, 3, 4, 5, 6, 7, 8]
    results = ResultList(name="QSVT", results=[])
    for kappa in kappa_list:
        qsvt_solver = LSESolver(A, b, simulator, kappa=kappa, shot=shot)
        x_solution, qsvt_circuit = qsvt_solver.solve_linear_system_quantum(
            statevector=False
        )
        depth = calculate_qc_depth(qsvt_circuit)
        classical_solution = np.linalg.solve(A, b)
        error = np.linalg.norm(x_solution - classical_solution) / np.linalg.norm(
            classical_solution
        )
        results.results.append(Result(depth, error))

    output_dir = "output/depth_matched" if use_depth_matched_dir else "output"
    save_result_list(results, f"{output_dir}/qsvt_{title}.json")
    return results


def collect_dfvqls_results(
    A: np.ndarray,
    b: np.ndarray,
    title: str,
    simulator: AerSimulator,
    shot: int,
    qsvt_depth: int = None,
    use_preconditioning: bool = False,
) -> ResultList:
    """
    Collect DF-VQLS results, optionally matching QSVT depth budget.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        title: Title for saving results
        qsvt_depth: If provided, run DF-VQLS until total depth matches this value
        use_preconditioning: If True, apply Jacobi preconditioning (improves convergence)
    """
    results = ResultList(
        name="DF-VQLS" + (" (Precond)" if use_preconditioning else ""), results=[]
    )

    # Apply preconditioning if requested
    if use_preconditioning:
        A_work, b_work, D_sqrt_inv = jacobi_precondition(A, b)
        print_condition_number_analysis(A, A_work, "K")
    else:
        A_work, b_work, D_sqrt_inv = A, b, None

    # First, determine DF-VQLS per-iteration depth with a quick run
    print("DF-VQLS: Calculating per-iteration circuit depth...")
    temp_solver = DFVQLSSolver(
        matrix_size=len(A),
        num_layers=2,
        optimizer_method="COBYLA",
        max_iter=1,
        verbose=False,
        simulator=simulator,
        shots=shot,
    )
    _, _, (num_circuit, den_circuit) = temp_solver.solve(A_work, b_work)
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
        checkpoint_iters = [max_iter // 10, max_iter // 5, max_iter // 2, max_iter]
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
        simulator=simulator,
        shots=shot,
    )

    print(f"\nDF-VQLS: Running optimization with max_iter={max_iter}")
    x_solution, metadata, (num_circuit, den_circuit) = dfvqls_solver.solve(
        A_work, b_work, track_iterations=True
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

    # Extract results at EVERY iteration instead of just checkpoints
    print(f"DF-VQLS: Processing {len(iteration_history)} iterations...")
    for iter_data in iteration_history:
        actual_iter = iter_data["iteration"] + 1  # Convert 0-indexed to 1-indexed

        # Reconstruct solution from parameters at this iteration
        iter_params = iter_data["params"]
        x_iter = dfvqls_solver.get_solution_at_params(iter_params, A_work, b_work)

        # Recover original solution if preconditioning was used
        if use_preconditioning:
            x_iter = recover_solution(x_iter, D_sqrt_inv)

        # Calculate error at this iteration
        error = np.linalg.norm(x_iter - classical_solution) / np.linalg.norm(
            classical_solution
        )

        # Store result (depth × actual_iteration for fair comparison)
        results.results.append(Result(depth * actual_iter, error))

        # Print progress every 50 iterations or at the end
        if actual_iter % 50 == 0 or actual_iter == len(iteration_history):
            print(
                f"  iter={actual_iter}, depth={depth * actual_iter}, "
                f"error={error:.4f}, cost={iter_data['cost']:.6f}"
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
    result_list1: ResultList,
    result_list2: ResultList,
    title: str,
    use_depth_matched_dir: bool = False,
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
    shot = 8192
    kappa = 4
    load_dotenv()
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=os.getenv("API_KEY"),
        instance=os.getenv("CRN"),
        overwrite=True,
    )

    has_noise_list = [False, True]
    for has_noise in has_noise_list:
        if has_noise:
            service = QiskitRuntimeService()
            real_backend = service.backend("ibm_torino")
            simulator = AerSimulator.from_backend(real_backend)
        else:
            simulator = AerSimulator(method="statevector")

        # 2x2 system
        print("=" * 80)
        print("Testing 2x2 system")
        print("=" * 80)
        title = f"2x2_kappa={kappa}_noise={has_noise}"
        A, b = create_matrix_with_condition_number(2, kappa)
        qsvt_results = collect_qsvt_results(
            A, b, title, simulator, shot, use_depth_matched_dir=True
        )
        # Get max depth from QSVT results
        qsvt_max_depth = max(result.depth for result in qsvt_results.results)
        print(f"\nQSVT maximum depth: {qsvt_max_depth}")
        print("Running DF-VQLS with matched depth budget (with preconditioning)...\n")
        dfvqls_results = collect_dfvqls_results(
            A,
            b,
            title,
            simulator,
            shot,
            qsvt_depth=qsvt_max_depth,
            use_preconditioning=True,
        )
        draw_result_plot(
            qsvt_results, dfvqls_results, title, use_depth_matched_dir=True
        )

        # 4x4 system
        print("\n" + "=" * 80)
        print("Testing 4x4 system")
        print("=" * 80)
        title = f"4x4_kappa={kappa}_noise={has_noise}"
        A, b = create_matrix_with_condition_number(4, kappa)
        qsvt_results = collect_qsvt_results(
            A, b, title, simulator, shot, use_depth_matched_dir=True
        )
        qsvt_max_depth = max(result.depth for result in qsvt_results.results)
        print(f"\nQSVT maximum depth: {qsvt_max_depth}")
        print("Running DF-VQLS with matched depth budget (with preconditioning)...\n")
        dfvqls_results = collect_dfvqls_results(
            A,
            b,
            title,
            simulator,
            shot,
            qsvt_depth=qsvt_max_depth,
            use_preconditioning=True,
        )
        draw_result_plot(
            qsvt_results, dfvqls_results, title, use_depth_matched_dir=True
        )

        # 8x8 system
        print("\n" + "=" * 80)
        print("Testing 8x8 system")
        print("=" * 80)
        title = f"8x8_kappa={kappa}_noise={has_noise}"
        A, b = create_matrix_with_condition_number(8, kappa)
        qsvt_results = collect_qsvt_results(
            A, b, title, simulator, shot, use_depth_matched_dir=True
        )
        qsvt_max_depth = max(result.depth for result in qsvt_results.results)
        print(f"\nQSVT maximum depth: {qsvt_max_depth}")
        print("Running DF-VQLS with matched depth budget (with preconditioning)...\n")
        dfvqls_results = collect_dfvqls_results(
            A,
            b,
            title,
            simulator,
            shot,
            qsvt_depth=qsvt_max_depth,
            use_preconditioning=True,
        )
        draw_result_plot(
            qsvt_results, dfvqls_results, title, use_depth_matched_dir=True
        )

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
