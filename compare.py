import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

# Add src/qsvt to Python path for inverse_matrix import
sys.path.insert(0, str(Path(__file__).parent / "src" / "qsvt"))

from src.qsvt.lse_solver import LSESolver
from test_combined import create_problem_2x2, create_problem_4x4, create_problem_8x8


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
    kappa_list = [1, 2, 3, 4]
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
    return collect_qsvt_results(A, b)


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
    plt.savefig(f"{title}.pdf")
    plt.close()


def main() -> None:
    title = "2x2"
    A, b, _ = create_problem_2x2()
    qsvt_results = collect_qsvt_results(A, b)
    dfvqls_results = collect_dfvqls_results(A, b)
    draw_result_plot(qsvt_results, dfvqls_results, title)

    title = "4x4"
    A, b, _ = create_problem_4x4()
    qsvt_results = collect_qsvt_results(A, b)
    dfvqls_results = collect_dfvqls_results(A, b)
    draw_result_plot(qsvt_results, dfvqls_results, title)

    title = "8x8"
    A, b, _ = create_problem_8x8()
    qsvt_results = collect_qsvt_results(A, b)
    dfvqls_results = collect_dfvqls_results(A, b)
    draw_result_plot(qsvt_results, dfvqls_results, title)

    return


if __name__ == "__main__":
    main()
