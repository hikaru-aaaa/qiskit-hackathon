"""
DF-VQLS Solver implementation

Decomposition-Free Variational Quantum Linear Solverを提供します。
"""

from typing import Tuple
import numpy as np
from scipy.optimize import OptimizeResult
from qiskit import QuantumCircuit

from .base import LinearSystemSolver
from ..vqls.generalized import DFVQLSSolver as GeneralizedDFVQLSSolver


class DFVQLSSolver(LinearSystemSolver):
    """
    DF-VQLS実装

    Generalized DF-VQLSをラップして、統一インターフェースを提供します。
    """

    def __init__(
        self,
        matrix_size: int,
        num_layers: int = 3,
        optimizer_method: str = "COBYLA",
        max_iter: int = 200,
        random_seed: int = None,
        verbose: bool = True,
        use_parallel: bool = False,
    ):
        """
        DF-VQLSソルバーを初期化

        Args:
            matrix_size: 行列サイズ（2のべき乗のみ）
            num_layers: アンサッツの層数
            optimizer_method: 最適化手法
            max_iter: 最大反復回数
            random_seed: ランダムシード
            verbose: 詳細出力
            use_parallel: 並列実行
        """
        self.solver = GeneralizedDFVQLSSolver(
            matrix_size=matrix_size,
            num_layers=num_layers,
            optimizer_method=optimizer_method,
            max_iter=max_iter,
            random_seed=random_seed,
            verbose=verbose,
            use_parallel=use_parallel,
        )

    def get_solution_at_params(self, params: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        指定されたパラメータで解を再構築

        Args:
            params: アンサッツパラメータ
            A: 係数行列
            b: 右辺ベクトル

        Returns:
            スケール済み解ベクトル
        """
        return self.solver.get_solution_at_params(params, A, b)

    def solve(
        self, A: np.ndarray, b: np.ndarray, initial_params: np.ndarray = None,
        track_iterations: bool = False
    ) -> Tuple[np.ndarray, dict, Tuple[QuantumCircuit, QuantumCircuit]]:
        """
        線形方程式系 Ax = b をDF-VQLSで解く

        Args:
            A: 係数行列 (N×N)
            b: 右辺ベクトル (N,)
            initial_params: 初期パラメータ (optional, warm start用)
            track_iterations: 反復履歴を記録するか (optional)

        Returns:
            Tuple of (解ベクトル, メタデータ, (numerator_circuit, denominator_circuit))
            track_iterations=Trueの場合、メタデータに'iteration_history'が含まれる
        """
        x_quantum, result, circuits = self.solver.solve(
            A, b, initial_params=initial_params, track_iterations=track_iterations
        )

        metadata = {
            "method": "DF-VQLS",
            "final_cost": result.fun,
            "iterations": result.nfev,
            "success": result.success,
            "optimize_result": result,
        }

        # Include iteration history if tracking was enabled
        if track_iterations and hasattr(result, 'iteration_history'):
            metadata["iteration_history"] = result.iteration_history

        return x_quantum, metadata, circuits
