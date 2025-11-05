"""
統一インターフェースの使用例

DF-VQLSとQSVTの両方を同じインターフェースで使用できます。
"""

import numpy as np
from .dfvqls_solver import DFVQLSSolver
try:
    from .qsvt_solver import QSVTSolver
    QSVT_AVAILABLE = True
except ImportError:
    QSVT_AVAILABLE = False
    print("QSVT solver is not available (QSVT modules not found)")


def example_compare_solvers():
    """DF-VQLSとQSVTの比較例"""
    print("=" * 70)
    print("Unified Interface Example: Comparing DF-VQLS and QSVT")
    print("=" * 70)
    
    # テスト用の2×2行列（QSVTは2×2のみサポート）
    A = np.array([[3, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    
    print("\n問題:")
    print(f"行列 A:\n{A}")
    print(f"ベクトル b: {b}")
    
    # 古典解
    x_classical = np.linalg.solve(A, b)
    print(f"\n古典解: {x_classical}")
    
    # DF-VQLSで解く
    print("\n" + "-" * 70)
    print("DF-VQLS Solver")
    print("-" * 70)
    dfvqls_solver = DFVQLSSolver(
        matrix_size=2,  # 2×2に調整
        num_layers=2,
        max_iter=50,
        verbose=True
    )
    x_dfvqls, metadata_dfvqls = dfvqls_solver.solve(A, b)
    print(f"\nDF-VQLS解: {x_dfvqls}")
    print(f"メタデータ: {metadata_dfvqls}")
    
    error_dfvqls = np.linalg.norm(x_dfvqls - x_classical) / np.linalg.norm(x_classical)
    print(f"相対誤差: {error_dfvqls:.6e}")
    
    # QSVTで解く（利用可能な場合）
    if QSVT_AVAILABLE:
        print("\n" + "-" * 70)
        print("QSVT Solver")
        print("-" * 70)
        qsvt_solver = QSVTSolver(
            matrix_size=2,
            poly_degree=100,
            kappa=20.0,
            verbose=True,
            use_statevector=True
        )
        x_qsvt, metadata_qsvt = qsvt_solver.solve(A, b)
        print(f"\nQSVT解: {x_qsvt}")
        print(f"メタデータ: {metadata_qsvt}")
        
        error_qsvt = np.linalg.norm(x_qsvt - x_classical) / np.linalg.norm(x_classical)
        print(f"相対誤差: {error_qsvt:.6e}")
        
        # 比較
        print("\n" + "=" * 70)
        print("Comparison")
        print("=" * 70)
        print(f"DF-VQLS 相対誤差: {error_dfvqls:.6e}")
        print(f"QSVT     相対誤差: {error_qsvt:.6e}")
        print("=" * 70)
    else:
        print("\nQSVT solver is not available (skipped)")
    
    print("=" * 70)


if __name__ == "__main__":
    example_compare_solvers()

