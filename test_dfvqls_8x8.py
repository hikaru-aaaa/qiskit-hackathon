"""
8×8行列専用のDF-VQLSテストコード

ランダムに生成した行列に対してDF-VQLSで問題を解きます。
"""

import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vqls.generalized.solver import DFVQLSSolver


def create_matrix_with_condition_number(
    n: int, kappa: float, symmetric: bool = True, random_seed: int = None
) -> tuple:
    """
    指定した条件数の行列を生成する

    Parameters:
    -----------
    n : int
        行列のサイズ (n×n)
    kappa : float
        条件数 (κ = σ_max / σ_min)
    symmetric : bool
        対称行列にするかどうか (デフォルト: True)
    random_seed : int, optional
        ランダムシード

    Returns:
    --------
    A : np.ndarray
        条件数がκの n×n 行列
    b : np.ndarray
        右辺ベクトル
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 特異値を生成：最大を1、最小を1/κに設定
    # 中間の特異値は対数的に配置
    if n == 1:
        singular_values = np.array([1.0])
    else:
        # 対数スケールで特異値を配置
        singular_values = np.logspace(0, -np.log10(kappa), n)

    # 対角行列を作成
    Sigma = np.diag(singular_values)

    # ランダムな直交行列を生成
    Q1, _ = np.linalg.qr(np.random.randn(n, n))

    if symmetric:
        # 対称行列の場合: A = Q * Σ * Q^T
        A = Q1 @ Sigma @ Q1.T
    else:
        # 非対称行列の場合: A = U * Σ * V^T (SVD)
        Q2, _ = np.linalg.qr(np.random.randn(n, n))
        A = Q1 @ Sigma @ Q2.T

    # 右辺ベクトルを生成
    b = np.array([1] + [0] * (n - 2) + [1], dtype=float)

    return A, b


def test_dfvqls_8x8(
    kappa: float = 4.0,
    num_layers: int = 3,
    max_iter: int = 200,
    random_seed: int = 42,
    verbose: bool = True,
    use_parallel: bool = False,
):
    """
    8×8行列に対してDF-VQLSをテストする

    Parameters:
    -----------
    kappa : float
        行列の条件数 (デフォルト: 4.0)
    num_layers : int
        アンサッツの層数 (デフォルト: 3)
    max_iter : int
        最大反復回数 (デフォルト: 200)
    random_seed : int
        ランダムシード (デフォルト: 42)
    verbose : bool
        詳細出力するかどうか (デフォルト: True)
    use_parallel : bool
        並列実行を使用するかどうか (デフォルト: False)
    """
    print("=" * 80)
    print("DF-VQLS 8×8 テスト")
    print("=" * 80)
    
    # ランダムな行列を生成
    print(f"\n条件数 κ = {kappa} のランダムな8×8行列を生成中...")
    A, b = create_matrix_with_condition_number(
        n=8, kappa=kappa, symmetric=True, random_seed=random_seed
    )
    
    # 実際の条件数を確認
    u, s, vh = np.linalg.svd(A)
    actual_kappa = s.max() / s.min()
    print(f"実際の条件数: {actual_kappa:.4f}")
    
    # 古典解を計算
    print("\n古典解を計算中...")
    x_classical = np.linalg.solve(A, b)
    print(f"古典解: {x_classical}")
    
    # DF-VQLSで解く
    print("\n" + "-" * 80)
    print("DF-VQLSで解く")
    print("-" * 80)
    print(f"設定:")
    print(f"  行列サイズ: 8×8")
    print(f"  アンサッツ層数: {num_layers}")
    print(f"  最大反復回数: {max_iter}")
    print(f"  ランダムシード: {random_seed}")
    print(f"  並列実行: {use_parallel}")
    print()
    
    start_time = time.time()
    
    solver = DFVQLSSolver(
        matrix_size=8,
        num_layers=num_layers,
        optimizer_method="COBYLA",
        max_iter=max_iter,
        random_seed=random_seed,
        verbose=verbose,
        use_parallel=use_parallel,
    )
    
    x_quantum, result, (num_circuit, den_circuit) = solver.solve(A, b)
    
    elapsed_time = time.time() - start_time
    
    # 結果を表示
    print("\n" + "-" * 80)
    print("結果")
    print("-" * 80)
    print(f"DF-VQLS解: {x_quantum}")
    print()
    
    # 誤差計算
    error = np.linalg.norm(x_quantum - x_classical) / np.linalg.norm(x_classical)
    residual = np.linalg.norm(A @ x_quantum - b)
    
    print(f"相対誤差: {error:.6e}")
    print(f"残差: {residual:.6e}")
    print(f"実行時間: {elapsed_time:.2f}秒")
    print(f"反復回数: {result.nfev}")
    print(f"最終コスト: {result.fun:.6e}")
    print(f"成功: {result.success}")
    
    # 回路の深さを計算（オプション）
    if verbose:
        try:
            from compare import calculate_qc_depth
            depth_num = calculate_qc_depth(num_circuit)
            depth_den = calculate_qc_depth(den_circuit)
            print(f"分子回路の深さ: {depth_num}")
            print(f"分母回路の深さ: {depth_den}")
        except:
            pass
    
    print("\n" + "=" * 80)
    
    # 精度評価
    if error < 1e-3:
        print("✓ 優秀な精度を達成しました！")
    elif error < 1e-2:
        print("✓ 良好な精度を達成しました！")
    elif error < 0.1:
        print("✓ 許容可能な精度を達成しました！")
    else:
        print("⚠ 中程度の精度 - より多くの反復または層数が必要かもしれません")
    
    print("=" * 80)
    
    return {
        "solution": x_quantum,
        "classical_solution": x_classical,
        "error": error,
        "residual": residual,
        "time": elapsed_time,
        "iterations": result.nfev,
        "cost": result.fun,
        "success": result.success,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="8×8行列専用のDF-VQLSテスト")
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="行列の条件数 (指定しない場合は1~4までループ)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="アンサッツの層数 (デフォルト: 3)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="最大反復回数 (デフォルト: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード (デフォルト: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="詳細出力を抑制",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="並列実行を有効化",
    )
    
    args = parser.parse_args()
    
    # kappaが指定されていない場合は1~4までループ
    if args.kappa is None:
        kappa_list = [1.0, 2.0, 3.0, 4.0]
        total_kappa = len(kappa_list)
        
        print("\n" + "=" * 80)
        print(f"kappa = 1~4 のシミュレーションを開始します (全{total_kappa}回)")
        print("=" * 80)
        
        results = []
        total_start_time = time.time()
        
        for idx, kappa in enumerate(kappa_list, 1):
            print("\n" + "=" * 80)
            print(f"[{idx}/{total_kappa}] kappa = {kappa} のシミュレーション開始")
            print("=" * 80)
            
            kappa_start_time = time.time()
            
            result = test_dfvqls_8x8(
                kappa=kappa,
                num_layers=args.layers,
                max_iter=args.max_iter,
                random_seed=args.seed,
                verbose=not args.quiet,
                use_parallel=args.parallel,
            )
            
            kappa_elapsed_time = time.time() - kappa_start_time
            
            if result:
                result["kappa"] = kappa
                results.append(result)
                
                print(f"\n[{idx}/{total_kappa}] kappa = {kappa} 完了")
                print(f"  実行時間: {kappa_elapsed_time:.2f}秒")
                print(f"  相対誤差: {result['error']:.6e}")
                print(f"  残差: {result['residual']:.6e}")
                print(f"  反復回数: {result['iterations']}")
                print(f"  最終コスト: {result['cost']:.6e}")
            else:
                print(f"\n[{idx}/{total_kappa}] kappa = {kappa} 失敗")
            
            # 残りの推定時間を計算
            if idx < total_kappa:
                avg_time_per_kappa = (time.time() - total_start_time) / idx
                remaining_kappa = total_kappa - idx
                estimated_remaining_time = avg_time_per_kappa * remaining_kappa
                print(f"  残り推定時間: {estimated_remaining_time:.2f}秒 ({remaining_kappa}回)")
        
        total_elapsed_time = time.time() - total_start_time
        
        # 全体の結果をまとめて表示
        print("\n" + "=" * 80)
        print("全体の結果サマリー")
        print("=" * 80)
        print(f"総実行時間: {total_elapsed_time:.2f}秒")
        print(f"平均実行時間: {total_elapsed_time / total_kappa:.2f}秒/回")
        print()
        print(f"{'kappa':<8} {'相対誤差':<15} {'残差':<15} {'反復回数':<10} {'最終コスト':<15} {'実行時間(秒)':<12}")
        print("-" * 80)
        
        for result in results:
            print(
                f"{result['kappa']:<8.1f} "
                f"{result['error']:<15.6e} "
                f"{result['residual']:<15.6e} "
                f"{result['iterations']:<10} "
                f"{result['cost']:<15.6e} "
                f"{result['time']:<12.2f}"
            )
        
        print("=" * 80)
    else:
        # 単一のkappaで実行
        test_dfvqls_8x8(
            kappa=args.kappa,
            num_layers=args.layers,
            max_iter=args.max_iter,
            random_seed=args.seed,
            verbose=not args.quiet,
            use_parallel=args.parallel,
        )

