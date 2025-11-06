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


def collect_qsvt_results(A: np.ndarray, b: np.ndarray) -> ResultList:
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
    return results


def collect_dfvqls_results(A: np.ndarray, b: np.ndarray) -> ResultList:
    max_iter_num = [100, 200, 300, 400, 500]
    results = ResultList(name="DF-VQLS", results=[])

    for max_iter in max_iter_num:
        # Create DF-VQLS solver with varying num_layers
        dfvqls_solver = DFVQLSSolver(
            matrix_size=len(A),
            num_layers=2,
            optimizer_method="COBYLA",
            max_iter=max_iter,
            verbose=False,
        )

        # Solve and get circuits
        x_solution, metadata, (num_circuit, den_circuit) = dfvqls_solver.solve(A, b)

        # Calculate max depth (critical path for parallel execution)
        depth_num = calculate_qc_depth(num_circuit)
        depth_den = calculate_qc_depth(den_circuit)
        depth = max(depth_num, depth_den)

        # Calculate error
        classical_solution = np.linalg.solve(A, b)
        error = np.linalg.norm(x_solution - classical_solution) / np.linalg.norm(
            classical_solution
        )

        results.results.append(Result(depth, error))
        print(
            f"DF-VQLS: num_layers={max_iter}, depth={depth * max_iter}, error={error:.4f}"
        )

    return results


def save_result_lists(
    result_list1: ResultList, result_list2: ResultList, filename: str
) -> None:
    """Save ResultList data to JSON file for later access."""
    data = {
        "result_list1": asdict(result_list1),
        "result_list2": asdict(result_list2),
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filename}")


def load_result_lists(filename: str) -> tuple[ResultList, ResultList]:
    """Load ResultList data from JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    result_list1 = ResultList(
        name=data["result_list1"]["name"],
        results=[Result(**r) for r in data["result_list1"]["results"]],
    )
    result_list2 = ResultList(
        name=data["result_list2"]["name"],
        results=[Result(**r) for r in data["result_list2"]["results"]],
    )
    return result_list1, result_list2


def draw_result_plot(
    result_list1: ResultList, result_list2: ResultList, title: str
) -> None:
    data_filename = f"output/{title}_data.json"
    save_result_lists(result_list1, result_list2, data_filename)

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
    title = "2x2"
    A, b = create_matrix_with_condition_number(2, kappa)
    qsvt_results = collect_qsvt_results(A, b)
    dfvqls_results = collect_dfvqls_results(A, b)
    draw_result_plot(qsvt_results, dfvqls_results, title)

    title = "4x4"
    A, b = create_matrix_with_condition_number(4, kappa)
    qsvt_results = collect_qsvt_results(A, b)
    dfvqls_results = collect_dfvqls_results(A, b)
    draw_result_plot(qsvt_results, dfvqls_results, title)

    title = "8x8"
    A, b = create_matrix_with_condition_number(8, kappa)
    qsvt_results = collect_qsvt_results(A, b)
    dfvqls_results = collect_dfvqls_results(A, b)
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
