"""
DF-VQLS テスト実行スクリプト

4×4システムでDF-VQLSアルゴリズムをテスト実行します。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vqls.dfvqls_4x4 import solve_dfvqls_4x4
import numpy as np


def test_dfvqls_quick():
    """クイックテスト: 少ない反復回数で動作確認"""
    print("\n" + "="*70)
    print("DF-VQLS クイックテスト (4×4システム)")
    print("="*70 + "\n")
    
    # テスト用の三重対角行列
    K = np.array([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]
    ], dtype=float)
    
    f = np.array([1, 0, 0, 1], dtype=float)
    
    # 古典解を計算
    u_classical = np.linalg.solve(K, f)
    print("古典解:")
    print(u_classical)
    print()
    
    # 量子解を計算（反復回数を減らしてクイックテスト）
    print("DF-VQLSで最適化中...")
    u_quantum, result = solve_dfvqls_4x4(
        K, f,
        max_iter=50,      # クイックテスト用に反復回数を減らす
        num_layers=2,     # レイヤー数を減らす
        seed=42,
        verbose=True
    )
    
    print("\nDF-VQLS解:")
    print(u_quantum)
    
    # 誤差を計算
    error = np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical)
    print(f"\n{'='*70}")
    print(f"相対誤差: {error:.6e}")
    print(f"最終コスト: {result.fun:.6f}")
    print(f"反復回数: {result.nfev}")
    
    if error < 0.1:
        print("✓ 優秀 - アルゴリズムは正常に動作しています！")
    elif error < 0.5:
        print("✓ 良好 - より多くの反復で改善する可能性があります")
    else:
        print("⚠ 誤差が大きい - パラメータ調整が必要かもしれません")
    print("="*70)


def test_dfvqls_full():
    """フルテスト: より多くの反復回数で高精度な結果を取得"""
    print("\n" + "="*70)
    print("DF-VQLS フルテスト (4×4システム)")
    print("="*70 + "\n")
    
    # テスト用の三重対角行列
    K = np.array([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]
    ], dtype=float)
    
    f = np.array([1, 0, 0, 1], dtype=float)
    
    # 古典解を計算
    u_classical = np.linalg.solve(K, f)
    print("古典解:")
    print(u_classical)
    print()
    
    # 量子解を計算
    print("DF-VQLSで最適化中...")
    u_quantum, result = solve_dfvqls_4x4(
        K, f,
        max_iter=200,     # より多くの反復回数
        num_layers=3,     # より表現力の高いアンザッツ
        seed=42,
        verbose=True
    )
    
    print("\nDF-VQLS解:")
    print(u_quantum)
    
    # 誤差を計算
    error = np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical)
    print(f"\n{'='*70}")
    print(f"相対誤差: {error:.6e}")
    print(f"最終コスト: {result.fun:.6f}")
    print(f"反復回数: {result.nfev}")
    
    if error < 0.01:
        print("✓ 非常に優秀 - アルゴリズムは高精度で動作しています！")
    elif error < 0.1:
        print("✓ 優秀 - アルゴリズムは正常に動作しています！")
    elif error < 0.5:
        print("✓ 良好 - より多くの反復で改善する可能性があります")
    else:
        print("⚠ 誤差が大きい - パラメータ調整が必要かもしれません")
    print("="*70)


def main():
    """メイン関数: テストモードを選択"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DF-VQLS テスト実行")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="テストモード: quick (クイックテスト) または full (フルテスト)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        test_dfvqls_quick()
    else:
        test_dfvqls_full()


if __name__ == "__main__":
    main()
