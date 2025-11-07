"""
QSVTとDF-VQLSの比較（平均化版）

2×2と4×4のVQLSについては、10回実行して平均を取ることで、
探索の凸凹を平滑化します。
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

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
    """QSVTの結果を収集（既存の結果を使用）"""
    # QSVTは既存の結果を使用（depth_matchedディレクトリから）
    output_dir = "output/depth_matched" if use_depth_matched_dir else "output"
    qsvt_file = f"{output_dir}/qsvt_{title}.json"
    
    # 既存の結果を読み込む
    if Path(qsvt_file).exists():
        print(f"QSVT: 既存の結果を使用: {qsvt_file}")
        with open(qsvt_file, "r") as f:
            data = json.load(f)
        return ResultList(
            name=data["name"],
            results=[Result(**r) for r in data["results"]]
        )
    else:
        # 既存の結果がない場合は新規に計算
        print(f"QSVT: 新規に計算します")
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
        
        save_result_list(results, qsvt_file)
        return results


def collect_dfvqls_results_single_run(
    A: np.ndarray,
    b: np.ndarray,
    qsvt_depth: int,
    use_preconditioning: bool,
    optimizer_method: str,
    per_iter_depth: int,
    max_iter: int,
    run_seed: int
) -> list[Result]:
    """
    単一のVQLS実行を行い、結果を返す
    
    Returns:
        List of Result objects (depth, error) for each iteration
    """
    # Apply preconditioning if requested
    if use_preconditioning:
        A_work, b_work, D_sqrt_inv = jacobi_precondition(A, b)
    else:
        A_work, b_work, D_sqrt_inv = A, b, None
    
    # Run optimization with iteration tracking enabled
    dfvqls_solver = DFVQLSSolver(
        matrix_size=len(A),
        num_layers=3,
        optimizer_method=optimizer_method,
        max_iter=max_iter,
        random_seed=run_seed,
        verbose=False,
    )
    
    # ランダム初期化を使用
    num_params = dfvqls_solver.solver.ansatz.num_parameters()
    np.random.seed(run_seed)
    initial_params = np.random.uniform(0, 2 * np.pi, num_params)
    
    x_solution, metadata, (num_circuit, den_circuit) = dfvqls_solver.solve(
        A_work, b_work, initial_params=initial_params, track_iterations=True
    )
    
    depth = per_iter_depth
    
    # Get classical solution for error calculation
    classical_solution = np.linalg.solve(A, b)
    
    # Extract iteration history
    iteration_history = metadata.get("iteration_history", [])
    
    # Extract results at EVERY iteration
    run_results = []
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
        run_results.append(Result(depth * actual_iter, error))
        
        # Print progress every 100 iterations
        if actual_iter % 100 == 0:
            print(f"    iter={actual_iter}, depth={depth * actual_iter}, error={error:.6f}, cost={iter_data['cost']:.6f}")
    
    return run_results


def collect_dfvqls_results_averaged(
    A: np.ndarray,
    b: np.ndarray,
    title: str,
    qsvt_depth: int = None,
    use_preconditioning: bool = False,
    optimizer_method: str = "BFGS",
    num_runs: int = 1
) -> ResultList:
    """
    Collect DF-VQLS results with averaging over multiple runs.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        title: Title for saving results
        qsvt_depth: If provided, run DF-VQLS until total depth matches this value
        use_preconditioning: If True, apply Jacobi preconditioning (improves convergence)
        optimizer_method: Optimization method ('BFGS', 'COBYLA', 'L-BFGS-B', etc.)
        num_runs: Number of runs to average over (default: 1, no averaging)
    """
    results = ResultList(name="DF-VQLS" + (" (Precond)" if use_preconditioning else ""), results=[])
    
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
        num_layers=3,
        optimizer_method=optimizer_method,
        max_iter=1,
        verbose=False,
    )
    _, _, (num_circuit, den_circuit) = temp_solver.solve(A_work, b_work)
    depth_num = calculate_qc_depth(num_circuit)
    depth_den = calculate_qc_depth(den_circuit)
    per_iter_depth = max(depth_num, depth_den)
    print(f"  Per-iteration depth: {per_iter_depth}")
    
    # Determine max_iter based on depth matching
    if qsvt_depth is not None:
        max_iter = qsvt_depth // per_iter_depth
        print(f"  QSVT depth budget: {qsvt_depth}")
        print(f"  Running DF-VQLS for {max_iter} iterations to match QSVT depth")
    else:
        max_iter = 500
        print(f"DF-VQLS: Running with default max_iter={max_iter}")
    
    # Run multiple times and collect results
    print(f"\nDF-VQLS: Running {num_runs} runs for averaging...")
    print(f"  Optimizer: {optimizer_method}")
    
    all_runs_results = []  # List of lists: each inner list is results from one run
    
    for run_idx in range(num_runs):
        run_seed = 42 + run_idx  # Different seed for each run
        print(f"  Run {run_idx + 1}/{num_runs}...")
        
        run_results = collect_dfvqls_results_single_run(
            A, b, qsvt_depth, use_preconditioning, optimizer_method,
            per_iter_depth, max_iter, run_seed
        )
        
        all_runs_results.append(run_results)
        print(f"  Run {run_idx + 1}/{num_runs} completed ({len(run_results)} iterations)")
    
    # Average results across runs
    # Group by depth and calculate average error
    depth_to_errors = defaultdict(list)
    
    for run_results in all_runs_results:
        for result in run_results:
            depth_to_errors[result.depth].append(result.error)
    
    # Calculate average for each depth
    averaged_results = []
    for depth in sorted(depth_to_errors.keys()):
        errors = depth_to_errors[depth]
        avg_error = np.mean(errors)
        averaged_results.append(Result(depth, avg_error))
    
    results.results = averaged_results
    
    print(f"\nDF-VQLS: Averaged {num_runs} runs, {len(averaged_results)} unique depths")
    
    # Save results to averaged subdirectory
    output_dir = "output/depth_matched/averaged" if qsvt_depth is not None else "output/averaged"
    save_result_list(results, f"{output_dir}/dfvqls_{title}_averaged.json")
    
    return results


def save_result_list(result_list: ResultList, filename: str) -> None:
    """Save a single ResultList to JSON file."""
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
    # 平均化した結果はaveragedディレクトリに保存
    output_dir = "output/depth_matched/averaged" if use_depth_matched_dir else "output/averaged"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/{title}.pdf")
    plt.close()


def main() -> None:
    kappa = 4
    
    # 2x2 system (10回平均)
    print("=" * 80)
    print("Testing 2x2 system (averaged over 10 runs)")
    print("=" * 80)
    title = f"2x2_kappa={kappa}"
    A, b = create_matrix_with_condition_number(2, kappa)
    qsvt_results = collect_qsvt_results(A, b, title, use_depth_matched_dir=True)
    qsvt_max_depth = max(result.depth for result in qsvt_results.results)
    print(f"\nQSVT maximum depth: {qsvt_max_depth}")
    print(f"Running DF-VQLS with matched depth budget (with preconditioning, 10 runs averaged)...\n")
    dfvqls_results = collect_dfvqls_results_averaged(
        A, b, title, qsvt_depth=qsvt_max_depth, use_preconditioning=True,
        optimizer_method="BFGS", num_runs=10
    )
    draw_result_plot(qsvt_results, dfvqls_results, title, use_depth_matched_dir=True)
    
    # 4x4 system (10回平均)
    print("\n" + "=" * 80)
    print("Testing 4x4 system (averaged over 10 runs)")
    print("=" * 80)
    title = f"4x4_kappa={kappa}"
    A, b = create_matrix_with_condition_number(4, kappa)
    qsvt_results = collect_qsvt_results(A, b, title, use_depth_matched_dir=True)
    qsvt_max_depth = max(result.depth for result in qsvt_results.results)
    print(f"\nQSVT maximum depth: {qsvt_max_depth}")
    print(f"Running DF-VQLS with matched depth budget (with preconditioning, 10 runs averaged)...\n")
    dfvqls_results = collect_dfvqls_results_averaged(
        A, b, title, qsvt_depth=qsvt_max_depth, use_preconditioning=True,
        optimizer_method="BFGS", num_runs=10
    )
    draw_result_plot(qsvt_results, dfvqls_results, title, use_depth_matched_dir=True)
    
    # 8x8 system (3回平均)
    print("\n" + "=" * 80)
    print("Testing 8x8 system (averaged over 3 runs)")
    print("=" * 80)
    title = f"8x8_kappa={kappa}"
    A, b = create_matrix_with_condition_number(8, kappa)
    qsvt_results = collect_qsvt_results(A, b, title, use_depth_matched_dir=True)
    qsvt_max_depth = max(result.depth for result in qsvt_results.results)
    print(f"\nQSVT maximum depth: {qsvt_max_depth}")
    print(f"Running DF-VQLS with matched depth budget (with preconditioning, 3 runs averaged)...\n")
    dfvqls_results = collect_dfvqls_results_averaged(
        A, b, title, qsvt_depth=qsvt_max_depth, use_preconditioning=True,
        optimizer_method="BFGS", num_runs=3
    )
    draw_result_plot(qsvt_results, dfvqls_results, title, use_depth_matched_dir=True)
    
    print("\n" + "=" * 80)
    print("All comparisons complete!")
    print("=" * 80)
    return


if __name__ == "__main__":
    main()

