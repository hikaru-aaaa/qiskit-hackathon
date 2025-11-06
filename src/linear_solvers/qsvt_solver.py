"""
QSVT Solver implementation

Quantum Singular Value Transformation (QSVT) を使用して
線形方程式系を解くソルバーを提供します。
"""

from typing import Tuple
import numpy as np
import warnings
import sys
from pathlib import Path

from .base import LinearSystemSolver

# Import LSESolver from qsvt module
try:
    # Ensure qsvt module can do relative imports
    qsvt_dir = Path(__file__).parent.parent / "qsvt"
    if str(qsvt_dir) not in sys.path:
        sys.path.insert(0, str(qsvt_dir))

    from ..qsvt.lse_solver import LSESolver
    QSVT_AVAILABLE = True
except ImportError as e:
    QSVT_AVAILABLE = False
    LSESolver = None


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
        if not QSVT_AVAILABLE:
            raise ImportError(
                "QSVT solver is not available. This may be due to missing dependencies (pyqsp, qiskit) "
                "or import issues with src.qsvt.lse_solver module."
            )

        # 元のqsvtブランチでは2×2限定のTODOがあったが、実装は任意サイズに対応可能
        # サイズが大きくなるとHadamardテストの計算コストが指数的に増加するが、
        # 高性能PCでのシミュレーションを想定し、制限は設けない
        # Hadamardテストの回数: 2^(n_qubits) 回
        # 2×2: 4回, 4×4: 16回, 8×8: 64回

        self.matrix_size = matrix_size
        self.verbose = verbose
        self.use_statevector = use_statevector
        self.poly_degree = poly_degree
        self.kappa = kappa
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        線形方程式系 Ax = b をQSVTで解く

        Args:
            A: 係数行列 (N×N)
            b: 右辺ベクトル (N,)

        Returns:
            Tuple of (解ベクトル, メタデータ)
        """
        import time

        # Create LSESolver instance for this problem
        solver = LSESolver(A, b)

        # Solve using QSVT
        t0 = time.time()
        x_quantum = solver.solve_linear_system_quantum(statevector=self.use_statevector)
        t_solve = time.time() - t0

        # Calculate error
        x_classical = np.linalg.solve(A, b)
        error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)

        # Build metadata
        metadata = {
            'method': 'QSVT',
            'time': t_solve,
            'error': error,
            'use_statevector': self.use_statevector,
            'matrix_size': self.matrix_size
        }

        return x_quantum, metadata

