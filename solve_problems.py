"""
Generalized DF-VQLSで実際の問題を解く

いくつかの異なる問題を解いてみましょう
"""

import sys
from pathlib import Path
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vqls.generalized import DFVQLSSolver


def problem_1_laplacian():
    """問題1: ラプラシアン行列（2次元の偏微分方程式の離散化）"""
    print("\n" + "=" * 70)
    print("問題1: ラプラシアン行列（8×8）")
    print("=" * 70)
    
    # ラプラシアン行列（2次元の離散ラプラシアン）
    # これは偏微分方程式の離散化でよく使われる
    K = np.zeros((8, 8))
    for i in range(8):
        K[i, i] = 4.0  # 対角要素
        if i > 0:
            K[i, i-1] = -1.0  # 下側
        if i < 7:
            K[i, i+1] = -1.0  # 上側
    
    # 右辺ベクトル（点源）
    f = np.zeros(8)
    f[0] = 1.0
    f[7] = 1.0
    
    print("\n行列 K:")
    print(K)
    print("\nベクトル f:")
    print(f)
    print()
    
    # 古典解
    x_classical = np.linalg.solve(K, f)
    print("古典解:")
    print(x_classical)
    print()
    
    # DF-VQLSで解く
    solver = DFVQLSSolver(
        matrix_size=8,
        num_layers=3,
        optimizer_method='COBYLA',
        max_iter=100,  # 適度な反復回数に調整
        random_seed=42,
        verbose=True,
        use_parallel=True
    )
    
    x_quantum, result = solver.solve(K, f)
    
    print("\nDF-VQLS解:")
    print(x_quantum)
    print()
    
    error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
    print(f"相対誤差: {error:.6e}")
    print(f"最終コスト: {result.fun:.6f}")
    print("=" * 70)


def problem_2_random_matrix():
    """問題2: ランダムな対称正定値行列"""
    print("\n" + "=" * 70)
    print("問題2: ランダムな対称正定値行列（8×8）")
    print("=" * 70)
    
    # ランダムな対称正定値行列を生成
    np.random.seed(123)
    A = np.random.rand(8, 8)
    K = A.T @ A + 0.1 * np.eye(8)  # 正定値にするために単位行列を加える
    
    # 右辺ベクトル
    f = np.random.rand(8)
    f = f / np.linalg.norm(f)  # 正規化
    
    print("\n行列 K (対称正定値):")
    print(K)
    print("\nベクトル f:")
    print(f)
    print()
    
    # 古典解
    x_classical = np.linalg.solve(K, f)
    print("古典解:")
    print(x_classical)
    print()
    
    # DF-VQLSで解く
    solver = DFVQLSSolver(
        matrix_size=8,
        num_layers=3,
        optimizer_method='COBYLA',
        max_iter=100,  # 適度な反復回数に調整
        random_seed=42,
        verbose=True,
        use_parallel=True
    )
    
    x_quantum, result = solver.solve(K, f)
    
    print("\nDF-VQLS解:")
    print(x_quantum)
    print()
    
    error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
    print(f"相対誤差: {error:.6e}")
    print(f"最終コスト: {result.fun:.6f}")
    print("=" * 70)


def problem_3_heat_equation():
    """問題3: 熱方程式の離散化（時間独立）"""
    print("\n" + "=" * 70)
    print("問題3: 熱方程式の離散化（8×8）")
    print("=" * 70)
    
    # 熱方程式の離散化行列
    # -u''(x) = f(x) の離散化
    h = 1.0 / 9.0  # ステップサイズ
    K = np.zeros((8, 8))
    
    for i in range(8):
        K[i, i] = 2.0 / (h**2)
        if i > 0:
            K[i, i-1] = -1.0 / (h**2)
        if i < 7:
            K[i, i+1] = -1.0 / (h**2)
    
    # 右辺ベクトル（正弦波）
    x_points = np.linspace(h, 1-h, 8)
    f = np.sin(np.pi * x_points)
    
    print("\n行列 K (熱方程式の離散化):")
    print(K)
    print("\nベクトル f (正弦波):")
    print(f)
    print()
    
    # 古典解
    x_classical = np.linalg.solve(K, f)
    print("古典解:")
    print(x_classical)
    print()
    
    # DF-VQLSで解く
    solver = DFVQLSSolver(
        matrix_size=8,
        num_layers=3,
        optimizer_method='COBYLA',
        max_iter=100,  # 適度な反復回数に調整
        random_seed=42,
        verbose=True,
        use_parallel=True
    )
    
    x_quantum, result = solver.solve(K, f)
    
    print("\nDF-VQLS解:")
    print(x_quantum)
    print()
    
    error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
    print(f"相対誤差: {error:.6e}")
    print(f"最終コスト: {result.fun:.6f}")
    print("=" * 70)


def problem_4_symmetric_tridiagonal():
    """問題4: 対称三重対角行列（より複雑な係数）"""
    print("\n" + "=" * 70)
    print("問題4: 対称三重対角行列（非一様係数）")
    print("=" * 70)
    
    # 非一様な三重対角行列
    K = np.zeros((8, 8))
    diag_values = [2.5, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7, 3.3]
    off_diag_values = [-0.8, -0.9, -0.7, -0.85, -0.75, -0.9, -0.8]
    
    for i in range(8):
        K[i, i] = diag_values[i]
        if i > 0:
            K[i, i-1] = off_diag_values[i-1]
            K[i-1, i] = off_diag_values[i-1]  # 対称性
    
    # 右辺ベクトル（階段関数）
    f = np.array([1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=float)
    
    print("\n行列 K (非一様三重対角):")
    print(K)
    print("\nベクトル f (階段関数):")
    print(f)
    print()
    
    # 古典解
    x_classical = np.linalg.solve(K, f)
    print("古典解:")
    print(x_classical)
    print()
    
    # DF-VQLSで解く
    solver = DFVQLSSolver(
        matrix_size=8,
        num_layers=3,
        optimizer_method='COBYLA',
        max_iter=100,  # 適度な反復回数に調整
        random_seed=42,
        verbose=True,
        use_parallel=True
    )
    
    x_quantum, result = solver.solve(K, f)
    
    print("\nDF-VQLS解:")
    print(x_quantum)
    print()
    
    error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
    print(f"相対誤差: {error:.6e}")
    print(f"最終コスト: {result.fun:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generalized DF-VQLSで問題を解く")
    parser.add_argument(
        "problem",
        choices=["1", "2", "3", "4", "all"],
        help="解く問題 (1: ラプラシアン, 2: ランダム行列, 3: 熱方程式, 4: 三重対角, all: すべて)"
    )
    args = parser.parse_args()
    
    if args.problem == "1":
        problem_1_laplacian()
    elif args.problem == "2":
        problem_2_random_matrix()
    elif args.problem == "3":
        problem_3_heat_equation()
    elif args.problem == "4":
        problem_4_symmetric_tridiagonal()
    elif args.problem == "all":
        problem_1_laplacian()
        problem_2_random_matrix()
        problem_3_heat_equation()
        problem_4_symmetric_tridiagonal()

