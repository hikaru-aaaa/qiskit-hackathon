# Generalized DF-VQLS 使い方ガイド

Generalized DF-VQLS (Decomposition-Free Variational Quantum Linear Solver) の使用方法を説明します。

## 概要

Generalized DF-VQLS は、任意の 2^n サイズの線形方程式系 `Ku = f` を解くための量子アルゴリズム実装です。モジュラー設計により、拡張性と再利用性を提供します。

## 基本的な使い方

### 1. インポート

```python
from src.vqls.generalized import DFVQLSSolver
import numpy as np
```

### 2. ソルバーの初期化

```python
solver = DFVQLSSolver(
    matrix_size=8,          # 行列サイズ (2^n のみ: 4, 8, 16, 32, ...)
    num_layers=3,           # アンサッツの層数 (デフォルト: 3)
    optimizer_method='COBYLA',  # 最適化手法 (デフォルト: 'COBYLA')
    max_iter=200,           # 最大反復回数 (デフォルト: 200)
    random_seed=42,         # ランダムシード (デフォルト: None)
    verbose=True,            # 詳細出力 (デフォルト: True)
    use_parallel=False      # 並列実行 (デフォルト: False)
)
```

### 3. 線形方程式を解く

```python
# 係数行列 K (8×8)
K = np.array([
    [2, -1,  0,  0,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0,  0,  0],
    [0, -1,  2, -1,  0,  0,  0,  0],
    [0,  0, -1,  2, -1,  0,  0,  0],
    [0,  0,  0, -1,  2, -1,  0,  0],
    [0,  0,  0,  0, -1,  2, -1,  0],
    [0,  0,  0,  0,  0, -1,  2, -1],
    [0,  0,  0,  0,  0,  0, -1,  2]
], dtype=float)

# 右辺ベクトル f (8要素)
f = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float)

# 解を計算
u_quantum, result = solver.solve(K, f)

# 結果
print("量子解:", u_quantum)
print("最終コスト:", result.fun)
print("反復回数:", result.nfev)
```

## パラメータ説明

### `matrix_size`

- **型**: `int`
- **説明**: 行列サイズ（2 のべき乗のみ）
- **例**: `4, 8, 16, 32, 64, ...`
- **必須**: はい

### `num_layers`

- **型**: `int`
- **説明**: ハードウェア効率アンサッツの層数
- **デフォルト**: `3`
- **推奨値**: 2-4 層（システムサイズに応じて）

### `optimizer_method`

- **型**: `str`
- **説明**: 最適化手法
- **デフォルト**: `'COBYLA'`
- **利用可能**: `'COBYLA'`, `'BFGS'`, `'L-BFGS-B'`, など（scipy.optimize.minimize でサポートされているもの）

### `max_iter`

- **型**: `int`
- **説明**: 最大反復回数
- **デフォルト**: `200`
- **推奨**: 100-500（問題の複雑さに応じて）

### `random_seed`

- **型**: `int` または `None`
- **説明**: 再現性のためのランダムシード
- **デフォルト**: `None`
- **推奨**: デバッグ時は固定値（例: `42`）を使用

### `verbose`

- **型**: `bool`
- **説明**: 詳細な進捗情報を表示
- **デフォルト**: `True`
- **効果**: プログレスバーと詳細な統計情報を表示

### `use_parallel`

- **型**: `bool`
- **説明**: 分子と分母回路の並列実行を有効化
- **デフォルト**: `False`
- **注意**: Python の GIL により、必ずしも高速化されるとは限りません

## 使用例

### 例 1: 8×8 三重対角行列

```python
from src.vqls.generalized import DFVQLSSolver
import numpy as np

# 三重対角行列を作成
def create_tridiagonal_matrix(size, diag=2.0, off_diag=-1.0):
    K = np.zeros((size, size))
    np.fill_diagonal(K, diag)
    np.fill_diagonal(K[1:], off_diag)
    np.fill_diagonal(K[:, 1:], off_diag)
    return K

K = create_tridiagonal_matrix(8, diag=2.0, off_diag=-1.0)
f = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float)

solver = DFVQLSSolver(
    matrix_size=8,
    num_layers=2,
    max_iter=100,
    verbose=True
)

u_quantum, result = solver.solve(K, f)

# 古典解と比較
u_classical = np.linalg.solve(K, f)
error = np.linalg.norm(u_quantum - u_classical) / np.linalg.norm(u_classical)
print(f"相対誤差: {error:.6e}")
```

### 例 2: 元の VQLS 問題

```python
from src.vqls.generalized import DFVQLSSolver
import numpy as np

# 係数セット
coefficient_set = [0.55, 0.225, 0.225]

# 行列を構築
a2 = coefficient_set[0] * np.eye(8)
a0 = coefficient_set[1] * np.diag([1, 1, -1, -1, 1, 1, -1, -1])
a1 = coefficient_set[2] * np.diag([1, 1, 1, 1, -1, -1, -1, -1])
A = a2 + a0 + a1

# ベクトル b (均等重ね合わせ)
b = np.array([1/np.sqrt(8)] * 8, dtype=float)

solver = DFVQLSSolver(
    matrix_size=8,
    num_layers=3,
    optimizer_method='COBYLA',
    max_iter=200,
    random_seed=42,
    verbose=True,
    use_parallel=True
)

x_quantum, result = solver.solve(A, b)

# フィデリティ計算
A_x_quantum = A.dot(x_quantum)
A_x_quantum_normalized = A_x_quantum / np.linalg.norm(A_x_quantum)
fidelity = (b.dot(A_x_quantum_normalized)) ** 2
print(f"フィデリティ: {fidelity:.6f}")
```

### 例 3: 並列実行の性能比較

```python
from src.vqls.generalized import DFVQLSSolver
import numpy as np
import time

K = create_tridiagonal_matrix(8)
f = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float)

# 逐次実行
solver_seq = DFVQLSSolver(
    matrix_size=8,
    max_iter=50,
    verbose=False,
    use_parallel=False
)
start = time.time()
u_seq, _ = solver_seq.solve(K, f)
time_seq = time.time() - start

# 並列実行
solver_par = DFVQLSSolver(
    matrix_size=8,
    max_iter=50,
    verbose=False,
    use_parallel=True
)
start = time.time()
u_par, _ = solver_par.solve(K, f)
time_par = time.time() - start

print(f"逐次実行: {time_seq:.2f}秒")
print(f"並列実行: {time_par:.2f}秒")
print(f"速度比: {time_seq/time_par:.2f}x")
```

## テストの実行

プロジェクトルートから以下のコマンドでテストを実行できます:

```bash
# クイックテスト (8×8, 10反復)
uv run python test_generalized.py quick

# フルテスト (8×8, 200反復)
uv run python test_generalized.py 8x8

# 16×16システム
uv run python test_generalized.py 16x16

# 並列実行の性能比較
uv run python test_generalized.py parallel

# 元のVQLS問題
uv run python test_generalized.py original
```

## プログレスバー

`verbose=True` に設定すると、以下のプログレスバーが表示されます:

1. **全体の最適化プログレスバー**: `Optimizing: X%|...|`

   - 最適化の全体進捗を表示

2. **各反復内のコスト計算プログレスバー**: `Cost computation: X%|...|`
   - 各反復でのコスト計算の 6 つのフェーズを表示:
     - Phase 1/6: Preparing states
     - Phase 2/6: Computing |u(θ)⟩
     - Phase 3/6: Building circuits
     - Phase 4/6: Running circuits
     - Phase 5/6: Running denominator circuit (逐次実行時のみ)
     - Phase 6/6: Computing cost

## 戻り値

`solve()` メソッドは以下のタプルを返します:

```python
u_quantum, result = solver.solve(K, f)
```

- **`u_quantum`**: `np.ndarray` - 量子アルゴリズムで得られた解ベクトル
- **`result`**: `scipy.optimize.OptimizeResult` - 最適化の結果
  - `result.fun`: 最終コスト値
  - `result.nfev`: 関数評価回数（反復回数）
  - `result.success`: 最適化の成功フラグ
  - `result.x`: 最適化されたパラメータ

## キャッシュ機能

StatePreparer は自動的にキャッシュを管理します:

- `vec(K)`: 行列のベクトル化結果をキャッシュ
- `vec(K^T)`: 転置行列のベクトル化結果をキャッシュ
- `f_norm`: 正規化ベクトルをキャッシュ

同じ行列・ベクトルで複数回コスト関数が呼ばれる場合、キャッシュにより高速化されます。

## トラブルシューティング

### エラー: "Matrix size must be a power of 2"

- **原因**: 行列サイズが 2 のべき乗ではない
- **解決**: `matrix_size` を `4, 8, 16, 32, ...` に設定

### エラー: "Matrix K must be N×N"

- **原因**: 行列のサイズが `matrix_size` と一致しない
- **解決**: 行列サイズを確認し、`matrix_size` と一致させる

### エラー: "Vector f must be N-dimensional"

- **原因**: ベクトルのサイズが `matrix_size` と一致しない
- **解決**: ベクトルサイズを確認し、`matrix_size` と一致させる

### プログレスバーが表示されない

- **原因**: `verbose=False` に設定されている、または出力がリダイレクトされている
- **解決**: `verbose=True` に設定し、ターミナルで直接実行

### 並列実行が遅い

- **原因**: Python の GIL やスレッドプールの初期化オーバーヘッド
- **解決**: `use_parallel=False` に設定して逐次実行を使用

## アーキテクチャ

Generalized DF-VQLS は以下のモジュールから構成されています:

- **`solver.py`**: メインソルバークラス `DFVQLSSolver`
- **`cost_function.py`**: コスト関数の計算
- **`circuit_builder.py`**: 量子回路の構築
- **`state_preparer.py`**: 量子状態の準備（キャッシュ機能付き）
- **`ansatz.py`**: 変分アンサッツ（ハードウェア効率アンサッツ）
- **`optimizer.py`**: 最適化ラッパー
- **`utils.py`**: ユーティリティ関数

詳細は `DESIGN.md` を参照してください。

## 参考資料

- **DF-VQLS 論文**: Yongchun Xu and Heng Hu, "Decomposition-free variational quantum linear solver: Application in computational mechanics" (2025)
- **設計ドキュメント**: `DESIGN.md`
- **最適化機会**: `OPTIMIZATION_OPPORTUNITIES.md`
