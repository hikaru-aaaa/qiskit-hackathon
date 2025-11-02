# QSVT Implementation in Qiskit

PennyLaneの`qml.qsvt`をQiskitで実装したバージョンです。

## ファイル構成

- `qsvt_qiskit.py`: QSVT実装のメインモジュール
- `intro_qsvt_qiskit.ipynb`: サンプルノートブック（PennyLane版の移植）

## 実装された機能

### 1. `block_encode(matrix, wires)`
行列をユニタリ演算子にブロックエンコードします。

**引数:**
- `matrix`: エンコードする行列（正方行列または長方行列）
- `wires`: 使用する量子ビットのインデックスリスト

**戻り値:**
- ブロックエンコーディングを含む`QuantumCircuit`

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

1. **角度計算**: `compute_qsvt_angles()`関数は簡略化された実装です。完全な実装には、多項式から位相角への変換アルゴリズムが必要です。

2. **ブロックエンコーディング**: QR分解を使用して行列をユニタリに補完していますが、より効率的な方法が存在する可能性があります。

3. **性能**: この実装は教育目的であり、大規模な行列には最適化が必要です。

## 依存関係

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib
```

## ライセンス

このコードはPennyLaneのQSVT実装を参考にしています。

