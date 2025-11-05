"""
統合テスト実行スクリプト

DF-VQLSとQSVTの両方のアルゴリズムで2×2, 4×4, 8×8の3つの問題を解き、比較します。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_combined import run_test


def main():
    """メイン関数"""
    run_test()


if __name__ == "__main__":
    main()
