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
    from src.linear_solvers import HybridSolver
    HYBRID_AVAILABLE = True
except ImportError as e:
    HYBRID_AVAILABLE = False

# pyqspのインストール確認
try:
    import pyqsp
    # pyqsp.angle_sequenceを事前にインポートして、モジュール属性として利用可能にする
    # これにより、inverse_matrix.pyでpyqsp.angle_sequence.QuantumSignalProcessingPhasesが
    # 正しく動作するようになる
    import pyqsp.angle_sequence
    import pyqsp.poly
except ImportError:
    print("⚠️  pyqspモジュールが見つかりません。")
    print("    QSVTソルバーを使用するには、以下のコマンドでインストールしてください:")
    print("    pip install pyqsp>=0.2.0")
    print("    または")
    print("    uv pip install pyqsp>=0.2.0")
    QSVT_AVAILABLE = False
else:
    try:
        # sys.pathを調整して相対インポートを解決
        import sys
        from pathlib import Path
        
        # qsvtディレクトリをパスに追加（相対インポート用）
        qsvt_dir = Path(__file__).parent / "src" / "qsvt"
        if str(qsvt_dir) not in sys.path:
            sys.path.insert(0, str(qsvt_dir))
        
        # qsvtモジュールを事前にインポート（inverse_matrix.pyのインポート用）
        # inverse_matrix.pyが「from qsvt import qsvt, transform_angles」を使っているため
        # qsvtをモジュールとして利用可能にする必要がある
        import importlib.util
        qsvt_module_path = qsvt_dir / "qsvt.py"
        spec = importlib.util.spec_from_file_location("qsvt", qsvt_module_path)
        qsvt_module = importlib.util.module_from_spec(spec)
        sys.modules["qsvt"] = qsvt_module
        spec.loader.exec_module(qsvt_module)
        
        # これでlse_solverをインポートできるはず
        from src.qsvt.lse_solver import LSESolver
        QSVT_AVAILABLE = True
    except ImportError as e:
        QSVT_AVAILABLE = False
        import traceback
        print(f"⚠️  QSVT solver is not available: {e}")
        print("詳細なエラー:")
        traceback.print_exc()
    except Exception as e:
        QSVT_AVAILABLE = False
        import traceback
        print(f"⚠️  QSVT solver is not available: {e}")
        print("詳細なエラー:")
        traceback.print_exc()


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


def solve_with_qsvt(A, b, matrix_size, use_statevector=True):
    """QSVTで解く（理論的には任意サイズに対応可能）"""
    if not QSVT_AVAILABLE:
        print("\n" + "-" * 70)
        print("QSVT Solver (利用不可)")
        print("-" * 70)
        return None
    
    # 元の実装では2×2限定のTODOがあったが、実装は任意サイズに対応可能
    # サイズが大きくなるとHadamardテストの計算コストが指数的に増加するが、
    # 高性能PCでのシミュレーションを想定し、制限は設けない
    
    print("\n" + "-" * 70)
    print(f"QSVT Solver ({'Statevector' if use_statevector else 'Measurement'})")
    print("-" * 70)
    
    try:
        start_time = time.time()
        # LSESolverは初期化時にAとbを受け取る
        solver = LSESolver(A, b)
        # solve_linear_system_quantum()で解を取得
        x_quantum = solver.solve_linear_system_quantum(statevector=use_statevector)
        elapsed_time = time.time() - start_time
        
        # 誤差計算
        x_classical = np.linalg.solve(A, b)
        error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
        residual = np.linalg.norm(A @ x_quantum - b)
        
        print(f"✅ QSVT解: {x_quantum}")
        print(f"   相対誤差: {error:.6e}")
        print(f"   残差: {residual:.6e}")
        print(f"   実行時間: {elapsed_time:.2f}秒")
        
        return {
            'solution': x_quantum,
            'error': error,
            'residual': residual,
            'time': elapsed_time
        }
    except Exception as e:
        print(f"❌ QSVTエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def solve_with_hybrid(A, b, matrix_size, max_iter=100):
    """ハイブリッドソルバーで解く"""
    if not HYBRID_AVAILABLE:
        print("\n" + "-" * 70)
        print("Hybrid Solver (利用不可)")
        print("-" * 70)
        return None
    
    print("\n" + "-" * 70)
    print("Hybrid Solver (QSVT + DF-VQLS)")
    print("-" * 70)
    
    try:
        start_time = time.time()
        solver = HybridSolver(
            matrix_size=matrix_size,
            qsvt_poly_degree=100,
            qsvt_kappa=20.0,
            qsvt_use_statevector=True,
            dfvqls_num_layers=3 if matrix_size >= 4 else 2,
            dfvqls_optimizer_method='COBYLA',
            dfvqls_max_iter=max_iter,
            dfvqls_random_seed=42,
            use_qsvt_initialization=True,
            verbose=False
        )
        x_quantum, metadata = solver.solve(A, b)
        elapsed_time = time.time() - start_time
        
        # 誤差計算
        x_classical = np.linalg.solve(A, b)
        error = metadata.get('final_error', np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical))
        residual = metadata.get('final_residual', np.linalg.norm(A @ x_quantum - b))
        
        print(f"✅ ハイブリッド解: {x_quantum}")
        print(f"   相対誤差: {error:.6e}")
        print(f"   残差: {residual:.6e}")
        print(f"   実行時間: {elapsed_time:.2f}秒")
        print(f"   QSVT使用: {metadata.get('qsvt_used', False)}")
        print(f"   反復回数: {metadata.get('dfvqls_result', {}).get('iterations', 'N/A')}")
        
        return {
            'solution': x_quantum,
            'error': error,
            'residual': residual,
            'time': elapsed_time,
            'qsvt_used': metadata.get('qsvt_used', False),
            'iterations': metadata.get('dfvqls_result', {}).get('iterations', 'N/A')
        }
    except Exception as e:
        print(f"❌ ハイブリッドエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_test():
    """全問題をテスト"""
    print("=" * 70)
    print("統合テスト: DF-VQLS vs QSVT vs Hybrid")
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
        
        # QSVTで解く
        result_qsvt = solve_with_qsvt(A, b, matrix_size, use_statevector=True)
        
        # ハイブリッドで解く
        result_hybrid = solve_with_hybrid(A, b, matrix_size, max_iter=max_iter)
        
        # 比較
        print("\n" + "-" * 70)
        print("比較")
        print("-" * 70)
        print(f"古典解: {x_classical}")
        if result_dfvqls:
            print(f"DF-VQLS: {result_dfvqls['solution']} (誤差: {result_dfvqls['error']:.6e})")
        if result_qsvt:
            print(f"QSVT:    {result_qsvt['solution']} (誤差: {result_qsvt['error']:.6e})")
        if result_hybrid:
            print(f"Hybrid:  {result_hybrid['solution']} (誤差: {result_hybrid['error']:.6e})")
        
        results.append({
            'problem': problem_name,
            'matrix_size': matrix_size,
            'dfvqls': result_dfvqls,
            'qsvt': result_qsvt,
            'hybrid': result_hybrid,
            'classical': x_classical
        })
    
    # サマリー
    print("\n" + "=" * 70)
    print("テスト結果サマリー")
    print("=" * 70)
    
    print(f"\n{'問題':<30} {'DF-VQLS誤差':<15} {'QSVT誤差':<15} {'Hybrid誤差':<15} {'DF-VQLS時間':<15} {'QSVT時間':<15} {'Hybrid時間':<15}")
    print("-" * 120)
    
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
        
        if result['hybrid']:
            hybrid_error = f"{result['hybrid']['error']:.6e}"
            hybrid_time = f"{result['hybrid']['time']:.2f}s"
        else:
            hybrid_error = "N/A"
            hybrid_time = "N/A"
        
        print(f"{problem_name:<30} {dfvqls_error:<15} {qsvt_error:<15} {hybrid_error:<15} {dfvqls_time:<15} {qsvt_time:<15} {hybrid_time:<15}")
    
    print("\n" + "=" * 70)
    print("✅ 統合テスト完了")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_test()

