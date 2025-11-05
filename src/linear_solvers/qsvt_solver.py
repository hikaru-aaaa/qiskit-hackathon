"""
QSVT Solver implementation

Quantum Singular Value Transformation (QSVT) を使用して
線形方程式系を解くソルバーを提供します。
"""

from typing import Tuple
import numpy as np

from .base import LinearSystemSolver
from ..qsvt.generalized import QSVTSolver as GeneralizedQSVTSolver


class QSVTSolver(LinearSystemSolver):
    """
    QSVT実装
    
    Generalized QSVTSolverをラップして、統一インターフェースを提供します。
    
    Note: 現在の実装は2×2行列に限定されています。
    より大きな行列への拡張は今後の課題です。
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
            matrix_size: 行列サイズ（現在は2のみサポート）
            poly_degree: 多項式近似の次数
            kappa: 条件数（condition number）の推定値
            random_seed: ランダムシード（未使用だが互換性のため）
            verbose: 詳細出力
            use_statevector: statevectorモードを使用（True）または測定モード（False）
            
        Raises:
            NotImplementedError: matrix_sizeが2以外の場合
        """
        if matrix_size != 2:
            raise NotImplementedError(
                f"QSVT solver currently supports only 2×2 matrices. "
                f"Got matrix_size={matrix_size}. "
                f"Generalization to larger matrices is in progress."
            )
        
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

