"""
統合テスト: DF-VQLSとQSVTの比較

2×2, 4×4, 8×8の3つの問題で両方のアルゴリズムをテストします。
"""

import sys
from pathlib import Path
import numpy as np
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.linear_solvers import DFVQLSSolver
try:
    from src.linear_solvers import QSVTSolver
    QSVT_AVAILABLE = True
except ImportError as e:
    QSVT_AVAILABLE = False
    print(f"⚠️  QSVT solver is not available: {e}")


def create_problem_2x2():
    """2×2問題: 対角優位な対称行列"""
    A = np.array([[3, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    return A, b, "2×2: 対角優位な対称行列"


def create_problem_4x4():
    """4×4問題: 三重対角行列"""
    A = np.array([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]
    ], dtype=float)
    b = np.array([1, 0, 0, 1], dtype=float)
    return A, b, "4×4: 三重対角行列"


def create_problem_8x8():
    """8×8問題: 三重対角行列"""
    A = np.zeros((8, 8), dtype=float)
    for i in range(8):
        A[i, i] = 2
        if i > 0:
            A[i, i-1] = -1
        if i < 7:
            A[i, i+1] = -1
    b = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    return A, b, "8×8: 三重対角行列"


def solve_with_dfvqls(A, b, matrix_size, max_iter=100):
    """DF-VQLSで解く"""
    print("\n" + "-" * 70)
    print("DF-VQLS Solver")
    print("-" * 70)
    
    try:
        start_time = time.time()
        solver = DFVQLSSolver(
            matrix_size=matrix_size,
            num_layers=3 if matrix_size >= 4 else 2,
            optimizer_method='COBYLA',
            max_iter=max_iter,
            random_seed=42,
            verbose=False,
            use_parallel=False
        )
        x_quantum, metadata = solver.solve(A, b)
        elapsed_time = time.time() - start_time
        
        # 誤差計算
        x_classical = np.linalg.solve(A, b)
        error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
        residual = np.linalg.norm(A @ x_quantum - b)
        
        print(f"✅ DF-VQLS解: {x_quantum}")
        print(f"   相対誤差: {error:.6e}")
        print(f"   残差: {residual:.6e}")
        print(f"   実行時間: {elapsed_time:.2f}秒")
        print(f"   反復回数: {metadata.get('iterations', 'N/A')}")
        print(f"   最終コスト: {metadata.get('final_cost', 'N/A'):.6f}")
        
        return {
            'solution': x_quantum,
            'error': error,
            'residual': residual,
            'time': elapsed_time,
            'iterations': metadata.get('iterations', 'N/A'),
            'cost': metadata.get('final_cost', 'N/A')
        }
    except Exception as e:
        print(f"❌ DF-VQLSエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def solve_with_qsvt(A, b, matrix_size):
    """QSVTで解く（理論的には任意サイズに対応可能）"""
    if not QSVT_AVAILABLE:
        print("\n" + "-" * 70)
        print("QSVT Solver (利用不可)")
        print("-" * 70)
        return None
    
    # 元の実装では2×2限定のTODOがあったが、実装は任意サイズに対応可能
    # ただし、4×4以上ではHadamardテストの計算コストが非常に高くなる
    # 4×4は試せるが、時間がかかる可能性がある
    if matrix_size > 4:
        print("\n" + "-" * 70)
        print(f"QSVT Solver ({matrix_size}×{matrix_size}は計算コストが高すぎるためスキップ)")
        print("-" * 70)
        return None
    
    print("\n" + "-" * 70)
    print("QSVT Solver (Statevector)")
    print("-" * 70)
    
    try:
        start_time = time.time()
        solver = QSVTSolver(
            matrix_size=matrix_size,
            poly_degree=100,
            kappa=20.0,
            verbose=False,
            use_statevector=True
        )
        x_quantum, metadata = solver.solve(A, b)
        elapsed_time = time.time() - start_time
        
        # 誤差計算
        x_classical = np.linalg.solve(A, b)
        error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
        residual = np.linalg.norm(A @ x_quantum - b)
        
        print(f"✅ QSVT解: {x_quantum}")
        print(f"   相対誤差: {error:.6e}")
        print(f"   残差: {residual:.6e}")
        print(f"   実行時間: {elapsed_time:.2f}秒")
        print(f"   多項式次数: {metadata.get('poly_degree', 'N/A')}")
        
        return {
            'solution': x_quantum,
            'error': error,
            'residual': residual,
            'time': elapsed_time,
            'poly_degree': metadata.get('poly_degree', 'N/A')
        }
    except Exception as e:
        print(f"❌ QSVTエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_test():
    """全問題をテスト"""
    print("=" * 70)
    print("統合テスト: DF-VQLS vs QSVT")
    print("=" * 70)
    print("\n2×2, 4×4, 8×8の3つの問題でテストします。")
    
    # 問題定義
    problems = [
        create_problem_2x2(),
        create_problem_4x4(),
        create_problem_8x8(),
    ]
    
    results = []
    
    for A, b, problem_name in problems:
        print("\n" + "=" * 70)
        print(f"問題: {problem_name}")
        print("=" * 70)
        print(f"行列 A ({A.shape[0]}×{A.shape[1]}):")
        print(A)
        print(f"ベクトル b: {b}")
        
        # 古典解
        x_classical = np.linalg.solve(A, b)
        print(f"\n古典解: {x_classical}")
        
        matrix_size = A.shape[0]
        max_iter = 100 if matrix_size <= 4 else 50  # 8×8は反復回数を減らす
        
        # DF-VQLSで解く
        result_dfvqls = solve_with_dfvqls(A, b, matrix_size, max_iter=max_iter)
        
        # QSVTで解く（2×2のみ）
        result_qsvt = solve_with_qsvt(A, b, matrix_size)
        
        # 比較
        print("\n" + "-" * 70)
        print("比較")
        print("-" * 70)
        print(f"古典解: {x_classical}")
        if result_dfvqls:
            print(f"DF-VQLS: {result_dfvqls['solution']} (誤差: {result_dfvqls['error']:.6e})")
        if result_qsvt:
            print(f"QSVT:    {result_qsvt['solution']} (誤差: {result_qsvt['error']:.6e})")
        
        results.append({
            'problem': problem_name,
            'matrix_size': matrix_size,
            'dfvqls': result_dfvqls,
            'qsvt': result_qsvt,
            'classical': x_classical
        })
    
    # サマリー
    print("\n" + "=" * 70)
    print("テスト結果サマリー")
    print("=" * 70)
    
    print(f"\n{'問題':<30} {'DF-VQLS誤差':<15} {'QSVT誤差':<15} {'DF-VQLS時間':<15} {'QSVT時間':<15}")
    print("-" * 90)
    
    for result in results:
        problem_name = result['problem']
        if result['dfvqls']:
            dfvqls_error = f"{result['dfvqls']['error']:.6e}"
            dfvqls_time = f"{result['dfvqls']['time']:.2f}s"
        else:
            dfvqls_error = "N/A"
            dfvqls_time = "N/A"
        
        if result['qsvt']:
            qsvt_error = f"{result['qsvt']['error']:.6e}"
            qsvt_time = f"{result['qsvt']['time']:.2f}s"
        else:
            qsvt_error = "N/A (未対応)"
            qsvt_time = "N/A"
        
        print(f"{problem_name:<30} {dfvqls_error:<15} {qsvt_error:<15} {dfvqls_time:<15} {qsvt_time:<15}")
    
    print("\n" + "=" * 70)
    print("✅ 統合テスト完了")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_test()

