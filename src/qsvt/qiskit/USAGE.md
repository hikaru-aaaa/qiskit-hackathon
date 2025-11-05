# 使い方ガイド

## どちらの実装を使うべきか？

### PennyLane互換版を使う場合（推奨）✨

```python
from qsvt_qiskit_pennylane import block_encode_pennylane as block_encode, pc_phase, qsvt
```

**メリット:**
- PennyLaneと完全互換
- 長方行列でより効率的（少ない量子ビット）
- notebookが動作する

**使うべき場合:**
- PennyLaneのコードを移植している
- 長方行列を扱う
- 量子ビット数を節約したい

### 標準版を使う場合

```python
from qsvt_qiskit import block_encode, pc_phase, qsvt
```

**メリット:**
- 標準的な数学的定義に忠実
- 教育目的に適している
- 理論的に理解しやすい

**使うべき場合:**
- 正方行列のみ扱う
- アルゴリズムの理論を学びたい
- 数学的な明確性を重視する

## エラーの対処法

### `ValueError: Matrix dimension 3 requires block encoding size 6...`

**原因:**
標準版の`block_encode`を使っていて、長方行列に対して量子ビットが不足している。

**解決策1（推奨）:** PennyLane互換版を使う
```python
from qsvt_qiskit_pennylane import block_encode_pennylane as block_encode
B = [[0.5, -0.5, 0.5]]
qc = block_encode(B, list(range(2)))  # 2 qubits でOK!
```

**解決策2:** より多くの量子ビットを使う
```python
from qsvt_qiskit import block_encode
B = [[0.5, -0.5, 0.5]]
qc = block_encode(B, list(range(3)))  # 3 qubits 必要
```

## 量子ビット要件の比較

| 行列 | 標準版 | PennyLane版 |
|------|--------|-------------|
| 2×2 | 2 qubits | 2 qubits |
| 3×3 | 3 qubits | 3 qubits |
| 1×3 | 3 qubits | **2 qubits** ✨ |
| 2×4 | 3 qubits | **3 qubits** |
| 4×4 | 3 qubits | 3 qubits |

**計算式:**
- **標準版**: `ceil(log2(2 × max(m, n)))` 量子ビット
- **PennyLane版**: `ceil(log2(m + n))` 量子ビット

## Jupyter Notebookでの使い方

### ケース1: `intro_qsvt_qiskit.ipynb`を実行

最初のセルが以下のようになっているか確認：
```python
from qsvt_qiskit_pennylane import block_encode_pennylane as block_encode, pc_phase, qsvt
```

もし以下のようになっていたら修正が必要：
```python
from qsvt_qiskit import block_encode, pc_phase, qsvt  # ❌ 長方行列でエラー
```

### ケース2: 新しいnotebookで使う

```python
import numpy as np
from qiskit.quantum_info import Operator

# PennyLane互換版をインポート（推奨）
from qsvt_qiskit_pennylane import block_encode_pennylane as block_encode

# 使い方は同じ
A = [[0.1, 0.2], [0.3, 0.4]]
qc = block_encode(A, list(range(2)))
U = Operator(qc).data
print(np.round(U, 2))
```

## トラブルシューティング

### Q: `ModuleNotFoundError: No module named 'qsvt_qiskit_pennylane'`

**A:** ファイルが同じディレクトリにあるか確認してください：
```bash
ls src/qsvt/qiskit/
# qsvt_qiskit.py
# qsvt_qiskit_pennylane.py  ← これが必要
```

### Q: notebookのカーネルを再起動したのにまだエラーが出る

**A:** セルの実行順序を確認してください。最初のセル（インポートのセル）を必ず最初に実行してください。

### Q: PennyLaneの出力と微妙に違う

**A:** 浮動小数点演算の精度による違いです。最大差分が0.01未満なら問題ありません。

## まとめ

- **迷ったらPennyLane互換版を使う** ✨
- 長方行列を扱うなら必ずPennyLane互換版
- notebookを動かすならPennyLane互換版
- 理論を学ぶなら標準版も価値がある

