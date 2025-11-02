# QSVT Implementation in Qiskit

PennyLaneの`qml.qsvt`をQiskitで実装したバージョンです。

## 🚀 クイックスタート

```python
# PennyLane互換版を使う（推奨）✨
from qsvt_qiskit_pennylane import block_encode_pennylane as block_encode
from qiskit.quantum_info import Operator
import numpy as np

# 正方行列（どちらの実装も同じ）
A = [[0.1, 0.2], [0.3, 0.4]]
qc = block_encode(A, list(range(2)))  # 2 qubits
U = Operator(qc).data

# 長方行列（PennyLane版は効率的！）
B = [[0.5, -0.5, 0.5]]
qc = block_encode(B, list(range(2)))  # 2 qubits (標準版は3 qubits必要)
```

**詳しい使い方は [`USAGE.md`](USAGE.md) を参照してください。**

## ファイル構成

- `qsvt_qiskit.py`: QSVT実装のメインモジュール（標準的なアルゴリズム）
- `qsvt_qiskit_pennylane.py`: **PennyLane互換実装** ✨（長方行列でより効率的）
- `intro_qsvt_qiskit.ipynb`: サンプルノートブック（PennyLane版の移植）
- `README_PENNYLANE.md`: PennyLane互換版の詳細説明
- **`USAGE.md`: 使い方ガイド（どちらを使うべきか）** 📖
- `DIFFERENCES.md`: 実装の違いの説明

## 実装された機能

### 1. `block_encode(matrix, wires)`
行列をユニタリ演算子にブロックエンコードします。

標準的なブロックエンコーディングスキームを使用：
```
U = [[A,              sqrt(I - A*A^†)  ],
     [sqrt(I - A^†*A), -A^†            ]]
```

**引数:**
- `matrix`: エンコードする行列（正方行列または長方行列）
- `wires`: 使用する量子ビットのインデックスリスト

**戻り値:**
- ブロックエンコーディングを含む`QuantumCircuit`

**注意:** 行列のノルムが1を超える場合は自動的に正規化されます。

### 2. `pc_phase(phi, dim, wires)`
Projector-Controlled Phase (PCPhase) ゲート。
最初の`dim`個の計算基底状態に位相`phi`を適用します。

**引数:**
- `phi`: 位相角
- `dim`: 部分空間の次元
- `wires`: 量子ビットのインデックスリスト

**戻り値:**
- PCPhase演算を含む`QuantumCircuit`

### 3. `qsvt(matrix_or_value, poly_coeffs, encoding_wires, block_encoding="embedding")`
Quantum Singular Value Transformation (QSVT)のメイン実装。

**引数:**
- `matrix_or_value`: 変換するスカラー値または行列
- `poly_coeffs`: 変換に使用する多項式の係数
- `encoding_wires`: エンコーディングに使用する量子ビット
- `block_encoding`: ブロックエンコーディングのタイプ（デフォルト: "embedding"）

**戻り値:**
- QSVTを実装する`QuantumCircuit`

### 4. `qsvt_matrix_transform(matrix, poly_coeffs, encoding_wires=None)`
行列の特異値を多項式に従って変換します。

**引数:**
- `matrix`: 入力行列
- `poly_coeffs`: 多項式係数
- `encoding_wires`: エンコーディング用の量子ビット（Noneの場合は自動決定）

**戻り値:**
- 変換された行列（出力ユニタリの左上ブロック）

## 使用例

```python
import numpy as np
from qsvt_qiskit import qsvt, block_encode, pc_phase
from qiskit.quantum_info import Operator

# 多項式係数の定義
target_poly = [0, -3 * 0.5, 0, 5 * 0.5]

# スカラー値に対するQSVT
a = 0.5
qc = qsvt(a, target_poly, encoding_wires=[0])
result = Operator(qc).data[0, 0]

# 2x2行列に対するブロックエンコーディング（2量子ビット）
A = [[0.1, 0.2], [0.3, 0.4]]
qc = block_encode(A, list(range(2)))  # 4x4 unitary
U = Operator(qc).data

# 1x3行列に対するブロックエンコーディング（3量子ビット必要）
B = [[0.5, -0.5, 0.5]]
qc = block_encode(B, list(range(3)))  # 8x8 unitary (not 4x4!)
U = Operator(qc).data

# 行列に対するQSVT
A = np.diag(np.linspace(-1, 1, 16))
qc = qsvt(A, target_poly, encoding_wires=list(range(5)))
U_out = Operator(qc).data
```

## PennyLaneとの対応

| PennyLane | Qiskit実装 |
|-----------|-----------|
| `qml.qsvt()` | `qsvt()` |
| `qml.BlockEncode()` | `block_encode()` |
| `qml.PCPhase()` | `pc_phase()` |
| `qml.matrix()` | `Operator().data` |

## 注意事項

1. **角度計算**: `compute_qsvt_angles()`関数は簡略化された実装です。完全な実装には、多項式から位相角への変換アルゴリズム（例：`poly_to_angles`）が必要です。

2. **ブロックエンコーディング**: 標準的なブロックエンコーディングスキームを使用していますが、PennyLaneの実装と異なる点があります：
   - **正方行列**: PennyLaneと完全一致（例：2x2行列 → 2量子ビット）
   - **長方行列**: PennyLaneは最適化されたエンコーディングを使用（例：1x3行列 → 2量子ビット）、我々の実装は標準的なスキームを使用（1x3行列 → 3量子ビット）
   - **量子ビット要件**: m×n行列の場合、max(m,n)次元の2倍のサイズが必要なので、⌈log₂(2×max(m,n))⌉量子ビットが必要です

3. **数値安定性**: 行列の固有値計算や平方根行列の計算において、数値誤差が生じる可能性があります。polar分解を使用してユニタリ性を保証していますが、完全な精度は保証されません。

4. **性能**: この実装は教育目的であり、大規模な行列には最適化が必要です。

## 依存関係

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib
```

## ライセンス

このコードはPennyLaneのQSVT実装を参考にしています。

