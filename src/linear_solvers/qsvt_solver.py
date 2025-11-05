"""
QSVT Solver implementation

Quantum Singular Value Transformation (QSVT) を使用して
線形方程式系を解くソルバーを提供します。
"""

from typing import Tuple
import numpy as np
import warnings

from .base import LinearSystemSolver
from ..qsvt.generalized import QSVTSolver as GeneralizedQSVTSolver


class QSVTSolver(LinearSystemSolver):
    """
    QSVT実装
    
    Generalized QSVTSolverをラップして、統一インターフェースを提供します。
    
    Note: 任意サイズの行列に対応可能。Hadamardテストの計算コストが
    指数的に増加するため、大規模問題では実行時間が長くなる可能性があります。
    """
    
    def __init__(
        self,
        matrix_size: int,
        poly_degree: int = 100,
        kappa: float = 20.0,
        random_seed: int = None,
        verbose: bool = True,
        use_statevector: bool = True
    ):
        """
        QSVTソルバーを初期化
        
        Args:
            matrix_size: 行列サイズ（任意サイズに対応）
            poly_degree: 多項式近似の次数
            kappa: 条件数（condition number）の推定値
            random_seed: ランダムシード（未使用だが互換性のため）
            verbose: 詳細出力
            use_statevector: statevectorモードを使用（True）または測定モード（False）
        """
        # 元のqsvtブランチでは2×2限定のTODOがあったが、実装は任意サイズに対応可能
        # サイズが大きくなるとHadamardテストの計算コストが指数的に増加するが、
        # 高性能PCでのシミュレーションを想定し、制限は設けない
        # Hadamardテストの回数: 2^(n_qubits) 回
        # 2×2: 4回, 4×4: 16回, 8×8: 64回
        
        self.solver = GeneralizedQSVTSolver(
            matrix_size=matrix_size,
            poly_degree=poly_degree,
            kappa=kappa,
            random_seed=random_seed,
            verbose=verbose,
            use_statevector=use_statevector
        )
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        線形方程式系 Ax = b をQSVTで解く
        
        Args:
            A: 係数行列 (N×N)
            b: 右辺ベクトル (N,)
            
        Returns:
            Tuple of (解ベクトル, メタデータ)
        """
        x_quantum, metadata = self.solver.solve(A, b)
        
        # メタデータに追加情報を追加
        metadata['use_statevector'] = self.solver.use_statevector
        
        return x_quantum, metadata

