"""
DF-VQLS Solver implementation

Decomposition-Free Variational Quantum Linear Solverを提供します。
"""

from typing import Tuple
import numpy as np
from scipy.optimize import OptimizeResult

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
        optimizer_method: str = 'COBYLA',
        max_iter: int = 200,
        random_seed: int = None,
        verbose: bool = True,
        use_parallel: bool = False
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
            use_parallel=use_parallel
        )
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        線形方程式系 Ax = b をDF-VQLSで解く
        
        Args:
            A: 係数行列 (N×N)
            b: 右辺ベクトル (N,)
            
        Returns:
            Tuple of (解ベクトル, メタデータ)
        """
        x_quantum, result = self.solver.solve(A, b)
        
        metadata = {
            'method': 'DF-VQLS',
            'final_cost': result.fun,
            'iterations': result.nfev,
            'success': result.success,
            'optimize_result': result
        }
        
        return x_quantum, metadata

