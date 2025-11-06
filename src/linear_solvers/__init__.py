"""
Linear System Solvers

統一されたインターフェースで様々な量子線形方程式ソルバーを提供します。
"""

from .base import LinearSystemSolver
from .dfvqls_solver import DFVQLSSolver

try:
    from .qsvt_solver import QSVTSolver
    from .hybrid_solver import HybridSolver
    __all__ = ['LinearSystemSolver', 'DFVQLSSolver', 'QSVTSolver', 'HybridSolver']
except ImportError:
    # QSVTモジュールが利用できない場合
    try:
        from .hybrid_solver import HybridSolver
        __all__ = ['LinearSystemSolver', 'DFVQLSSolver', 'HybridSolver']
    except ImportError:
        # HybridSolverも利用できない場合
        __all__ = ['LinearSystemSolver', 'DFVQLSSolver']

