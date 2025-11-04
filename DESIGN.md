# DF-VQLS 一般化設計書

## 設計方針

1. **既存コードの保持**: `dfvqls.py`, `dfvqls_4x4.py`は変更しない
2. **クラスベース設計**: 機能ごとにクラス化し、拡張性を確保
3. **モジュール分離**: 機能ごとにファイルを分け、保守性を向上
4. **サイズ一般化**: 8×8, 16×16に対応（2^nのサイズ）

## ディレクトリ構造

```
src/vqls/generalized/
├── __init__.py              # パッケージ初期化
├── solver.py                # メインのDFVQLSSolverクラス
├── circuit_builder.py       # 量子回路構築（分子・分母回路）
├── cost_function.py         # コスト関数計算
├── optimizer.py             # 最適化処理（最適化器の抽象化）
├── state_preparer.py        # 状態準備（振幅エンコーディング）
├── ansatz.py                # アンザッツ定義（ハードウェア効率型）
└── utils.py                 # ユーティリティ関数（ベクトル化など）
```

## クラス設計

### 1. `DFVQLSSolver` (solver.py)
**役割**: メインのソルバークラス。全体のオーケストレーション

```python
class DFVQLSSolver:
    def __init__(self, matrix_size: int, num_layers: int = 3, ...):
        # サイズの検証（2^nであることを確認）
        # コンポーネントの初期化
        self.circuit_builder = CircuitBuilder(...)
        self.cost_function = CostFunction(...)
        self.optimizer = Optimizer(...)
        ...
    
    def solve(self, K: np.ndarray, f: np.ndarray) -> Solution:
        # 1. 入力検証
        # 2. コスト関数の定義
        # 3. 最適化実行
        # 4. 解の抽出とスケーリング
        # 5. 結果の返却
```

### 2. `CircuitBuilder` (circuit_builder.py)
**役割**: 量子回路の構築（分子・分母回路）

```python
class CircuitBuilder:
    def __init__(self, n_qubits: int, ansatz: Ansatz, ...):
        self.n_qubits = n_qubits  # log2(N)
        self.ansatz = ansatz
        ...
    
    def build_numerator_circuit(self, vec_K, f, u_theta) -> QuantumCircuit:
        # 分子回路: |vec(K)⟩ vs |u(θ)⟩⊗|f⟩
    
    def build_denominator_circuit(self, vec_K, vec_KT, u_theta) -> QuantumCircuit:
        # 分母回路: |u(θ)⟩⊗|vec(K^T)⟩ vs |vec(K)⟩⊗|u(θ)⟩
```

### 3. `CostFunction` (cost_function.py)
**役割**: コスト関数の計算（スワップテストの実行）

```python
class CostFunction:
    def __init__(self, circuit_builder, simulator, ...):
        self.circuit_builder = circuit_builder
        self.simulator = simulator
    
    def compute(self, params: np.ndarray, K: np.ndarray, f: np.ndarray) -> float:
        # 1. |u(θ)⟩を計算
        # 2. 分子回路を実行してコストを計算
        # 3. 分母回路を実行してコストを計算
        # 4. CG(θ) = 1 - |⟨f|K|u(θ)⟩|² / ⟨u(θ)|K^T K|u(θ)⟩
        return cost
```

### 4. `Optimizer` (optimizer.py)
**役割**: 最適化処理の抽象化（将来的に複数の最適化器に対応）

```python
class Optimizer:
    def __init__(self, method: str = 'COBYLA', max_iter: int = 200, ...):
        self.method = method
        self.max_iter = max_iter
        ...
    
    def optimize(self, cost_function, initial_params) -> OptimizeResult:
        # 最適化の実行（COBYLA, BFGSなど）
        # プログレスバーの統合
```

### 5. `StatePreparer` (state_preparer.py)
**役割**: 状態準備（振幅エンコーディング）

```python
class StatePreparer:
    def prepare_vec_K(self, K: np.ndarray) -> Tuple[np.ndarray, float]:
        # 行列のベクトル化と正規化
    
    def prepare_vector(self, v: np.ndarray) -> np.ndarray:
        # ベクトルの正規化
```

### 6. `Ansatz` (ansatz.py)
**役割**: アンザッツの定義（ハードウェア効率型）

```python
class Ansatz:
    def __init__(self, num_qubits: int, num_layers: int = 3):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
    
    def apply(self, circ: QuantumCircuit, qubits: List[int], params: np.ndarray):
        # アンザッツの適用（RY回転 + CZエンタングルメント）
```

### 7. `utils.py`
**役割**: ユーティリティ関数

```python
def validate_matrix_size(size: int) -> bool:
    # 2^nであることを確認
    
def calculate_qubits(matrix_size: int) -> int:
    # log2(N)を計算
    
def vectorize_matrix(K: np.ndarray) -> Tuple[np.ndarray, float]:
    # 行列のベクトル化
```

## サイズ対応の設計

### 制約
- **8×8**: 3量子ビット（log₂(8) = 3）
- **16×16**: 4量子ビット（log₂(16) = 4）
- 将来的には2^nのサイズに対応可能

### 量子ビット数の計算
```python
def calculate_qubits(matrix_size: int) -> int:
    """Calculate number of qubits needed for matrix size"""
    import math
    n_qubits = int(math.ceil(math.log2(matrix_size)))
    if 2**n_qubits != matrix_size:
        raise ValueError(f"Matrix size must be power of 2, got {matrix_size}")
    return n_qubits
```

### 回路の量子ビット数
- **分子回路**: 1 (ancilla) + 2*n_qubits (vec(K)) + 2*n_qubits (u⊗f) = 4*n_qubits + 1
- **分母回路**: 1 (ancilla) + 4*n_qubits (u⊗vec(K^T) + vec(K)⊗u) = 4*n_qubits + 1

## 拡張性の考慮

### 1. 最適化器の拡張
- `Optimizer`クラスを抽象化し、複数の最適化器（COBYLA, BFGS, SPSAなど）に対応
- 将来的に量子最適化器にも対応可能

### 2. アンザッツの拡張
- `Ansatz`クラスを基底クラスとし、異なるアンザッツ（Hardware-efficient, QAOAなど）を実装可能

### 3. 実行時間の改善
- 回路のキャッシュ機能
- 並列実行のサポート
- 近似手法の導入（将来的に）

### 4. 状態準備の最適化
- 特殊構造行列（疎行列、Toeplitz行列など）に対する効率的な状態準備回路

## 実装の優先順位

1. **Phase 1**: 基本的なクラス構造と8×8システムの実装
2. **Phase 2**: 16×16システムへの拡張
3. **Phase 3**: 最適化とパフォーマンス改善

## 使用例

```python
from src.vqls.generalized import DFVQLSSolver

# 8×8システム
solver = DFVQLSSolver(matrix_size=8, num_layers=3)
solution, result = solver.solve(K, f)

# 16×16システム
solver = DFVQLSSolver(matrix_size=16, num_layers=4)
solution, result = solver.solve(K, f)
```

## 注意事項

1. **サイズ制限**: 現在は2^nのサイズのみ対応（8×8, 16×16など）
2. **シミュレータ専用**: 実量子デバイスでの実行は想定していない（振幅エンコーディングの制約）
3. **メモリ使用量**: 16×16で約2^16 = 65536次元の状態ベクトルが必要

