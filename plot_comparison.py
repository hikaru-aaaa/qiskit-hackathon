"""
QSVTとDF-VQLSの比較グラフを作成

既存のJSONファイルを読み込んで、depthが10000くらいまでの範囲で
わかりやすいグラフを作成します。
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_result_list(filename: str):
    """JSONファイルから結果を読み込む"""
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def plot_comparison(
    qsvt_file: str,
    dfvqls_file: str,
    output_file: str = "output/depth_matched/comparison_8x8_kappa4.pdf",
    max_depth: int = None,
    title: str = None
):
    """
    QSVTとDF-VQLSの比較グラフを作成
    
    Args:
        qsvt_file: QSVT結果のJSONファイルパス
        dfvqls_file: DF-VQLS結果のJSONファイルパス
        output_file: 出力ファイルパス
        max_depth: 表示する最大depth（Noneの場合はVQLSの最大depthまで）
        title: グラフのタイトル（Noneの場合は自動生成）
    """
    # データを読み込む
    qsvt_data = load_result_list(qsvt_file)
    dfvqls_data = load_result_list(dfvqls_file)
    
    # depthとerrorを抽出
    qsvt_depths = [r["depth"] for r in qsvt_data["results"]]
    qsvt_errors = [r["error"] for r in qsvt_data["results"]]
    
    dfvqls_depths = [r["depth"] for r in dfvqls_data["results"]]
    dfvqls_errors = [r["error"] for r in dfvqls_data["results"]]
    
    # max_depthが指定されていない場合は、VQLSを全て含み、かつQSVTはなるべく少ない領域を選択
    # VQLSの最大depthを超える最小のQSVTのdepthポイントを見つける
    if max_depth is None:
        if dfvqls_depths:
            vqls_max_depth = max(dfvqls_depths)
            # VQLSの最大depthを超える最小のQSVTのdepthポイントを見つける
            if qsvt_depths:
                # QSVTのdepthポイントをソート
                sorted_qsvt_depths = sorted(qsvt_depths)
                # VQLSの最大depthを超える最小のQSVTのdepthを探す
                for qsvt_depth in sorted_qsvt_depths:
                    if qsvt_depth >= vqls_max_depth:
                        max_depth = qsvt_depth
                        break
                # 見つからない場合は、QSVTの最大depthを使用
                if max_depth is None:
                    max_depth = max(qsvt_depths)
            else:
                max_depth = vqls_max_depth
        elif qsvt_depths:
            max_depth = max(qsvt_depths)
        else:
            max_depth = 10000
    
    # QSVTはmax_depthまでの範囲でフィルタリング
    qsvt_filtered = [
        (d, e) for d, e in zip(qsvt_depths, qsvt_errors) if d <= max_depth
    ]
    
    # DF-VQLSは全データポイントを表示（VQLSは全て含む）
    dfvqls_filtered = list(zip(dfvqls_depths, dfvqls_errors))
    
    if qsvt_filtered:
        qsvt_depths_filtered, qsvt_errors_filtered = zip(*qsvt_filtered)
    else:
        qsvt_depths_filtered, qsvt_errors_filtered = [], []
    
    if dfvqls_filtered:
        dfvqls_depths_filtered, dfvqls_errors_filtered = zip(*dfvqls_filtered)
    else:
        dfvqls_depths_filtered, dfvqls_errors_filtered = [], []
    
    # グラフを作成
    plt.figure(figsize=(10, 6))
    
    # QSVTのプロット（マーカー付き）
    if qsvt_depths_filtered:
        plt.plot(
            qsvt_depths_filtered,
            qsvt_errors_filtered,
            marker='o',
            markersize=8,
            linewidth=2,
            label=qsvt_data["name"],
            color='#2E86AB',
            markerfacecolor='#2E86AB',
            markeredgecolor='white',
            markeredgewidth=1.5
        )
    
    # DF-VQLSのプロット（線のみ、細め）
    if dfvqls_depths_filtered:
        plt.plot(
            dfvqls_depths_filtered,
            dfvqls_errors_filtered,
            marker='.',
            markersize=2,
            linewidth=1.5,
            label=dfvqls_data["name"],
            color='#A23B72',
            alpha=0.8
        )
    
    # タイトルを自動生成（指定されていない場合）
    if title is None:
        # ファイル名から情報を抽出
        if "2x2" in qsvt_file:
            size = "2×2"
        elif "4x4" in qsvt_file:
            size = "4×4"
        elif "8x8" in qsvt_file:
            size = "8×8"
        else:
            size = "N×N"
        
        if "kappa=1" in qsvt_file:
            kappa = "κ=1"
        elif "kappa=2" in qsvt_file:
            kappa = "κ=2"
        elif "kappa=3" in qsvt_file:
            kappa = "κ=3"
        elif "kappa=4" in qsvt_file:
            kappa = "κ=4"
        else:
            kappa = ""
        
        title = f"QSVT vs DF-VQLS Comparison ({size}, {kappa})"
    
    # グラフの設定
    plt.xlabel("Depth", fontsize=14, fontweight='bold')
    plt.ylabel("Relative Error", fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, max_depth)
    
    # 線形スケールで表示
    # plt.yscale('log')  # 対数スケールは使わない
    
    # レイアウトを調整
    plt.tight_layout()
    
    # 出力ディレクトリを作成
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"グラフを保存しました: {output_file}")
    
    # 統計情報を表示
    print("\n" + "=" * 80)
    print("統計情報")
    print("=" * 80)
    print(f"QSVT:")
    print(f"  データポイント数: {len(qsvt_depths_filtered)}")
    if qsvt_depths_filtered:
        print(f"  Depth範囲: {min(qsvt_depths_filtered)} - {max(qsvt_depths_filtered)}")
        print(f"  エラー範囲: {min(qsvt_errors_filtered):.6e} - {max(qsvt_errors_filtered):.6e}")
    print(f"\nDF-VQLS:")
    print(f"  データポイント数: {len(dfvqls_depths_filtered)}")
    if dfvqls_depths_filtered:
        print(f"  Depth範囲: {min(dfvqls_depths_filtered)} - {max(dfvqls_depths_filtered)}")
        print(f"  エラー範囲: {min(dfvqls_errors_filtered):.6e} - {max(dfvqls_errors_filtered):.6e}")
    print("=" * 80)
    
    plt.close()


if __name__ == "__main__":
    # 8×8, κ=4
    print("=" * 80)
    print("8×8, κ=4 の比較グラフを作成中...")
    print("=" * 80)
    plot_comparison(
        qsvt_file="output/depth_matched/qsvt_8x8_kappa=4.json",
        dfvqls_file="/Users/seitsubo/Downloads/depth_matched/dfvqls_8x8_kappa=4.json",
        output_file="output/depth_matched/comparison_8x8_kappa4.pdf",
        max_depth=None  # VQLSの最大depthまで
    )
    
    # 2×2, κ=3
    print("\n" + "=" * 80)
    print("2×2, κ=3 の比較グラフを作成中...")
    print("=" * 80)
    plot_comparison(
        qsvt_file="output/depth_matched/qsvt_2x2_kappa=3.json",
        dfvqls_file="output/depth_matched/dfvqls_2x2_kappa=3.json",
        output_file="output/depth_matched/comparison_2x2_kappa3.pdf",
        max_depth=None  # VQLSの最大depthまで
    )
    
    # 4×4, κ=3
    print("\n" + "=" * 80)
    print("4×4, κ=3 の比較グラフを作成中...")
    print("=" * 80)
    plot_comparison(
        qsvt_file="output/depth_matched/qsvt_4x4_kappa=3.json",
        dfvqls_file="output/depth_matched/dfvqls_4x4_kappa=3.json",
        output_file="output/depth_matched/comparison_4x4_kappa3.pdf",
        max_depth=None  # VQLSの最大depthまで
    )
    
    print("\n" + "=" * 80)
    print("すべての比較グラフを作成しました！")
    print("=" * 80)

