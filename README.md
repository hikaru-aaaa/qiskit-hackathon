# 参照資料

[Github](https://github.com/Qiskit/textbook/blob/main/notebooks/ch-applications/vqls.ipynb): IBM Qiskit TextbookのVQLS実装ノートブック なおdeprecated
[Qiita](https://qiita.com/Ayumu_walker/items/6b1131c6be64a8c5ad37): GitHubの和訳的だが関数定義が抜けてるし古い
[arxiv](https://arxiv.org/abs/1909.05820): 元論文

# qiita.py概要

- VQLS(Variational Quantum Linear Solver)の実装
- 参考資料のコードをQiskitの最新バージョンに対応させたもの

## 制約条件

簡易化のため、以下の制約を設けている

- 3量子bit系
- 求める式: $$Ax = b$$
  - A: I, Zで分解可能なもの
  - b: H|0> = |+++>
- 初期パラメタ、$$\frac{k}{1000}$$
- 最適化手法: COBYLA

## 既存コードからの変更点

- aer, assembleがdeprecatedになったため、qiskit_aerとAerSimulator.runへ変更

## メモ

`gate_type`はアダマールテストで分解した $$<\psi|\psi>$$ のI,Zに対応してる(らしい)。例えば、

```
[0, 0, 0] → I ⊗ I ⊗ I = I
[0, 1, 1] → I ⊗ Z ⊗ Z = Z₂Z₃
```

> The reason why we are applying two different "gate_types" is because this represents the pairs of gates shown in the expanded form of $$<\psi|\psi>$$.
> It is also important to note that for the purposes of this implementation (the systems of equations we will actually be solving, we are only concerned with the gates I and Z, so I only include support for these gates (The code includes number "identifiers" that signify the application of different gates, 0 for I and 1 for Z).

# タスク

- 理解していないこと
  - 何故アンザッツはその形を使用しているか
  - 一般の問題に対して行列Aをユニタリ行列の線形結合にどう分解するか - これは発表に必須なので、要調査
  - b = U|0>を満すようなUの探し方(今回はH⊗3だが一般には？)
- サブ目標
  - VQLSの回転パラメタを、はじめはクリフォードにすることでゲート分解が高速にできるか試したい
    - ソースは[これ](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.062413)のはず

- **適用しやすいモデルを探す**
  - 一旦最優先

# 1103進捗

- [DF-VQSLの論文](https://www.researchgate.net/publication/395543515_Decomposition-free_variational_quantum_linear_solver_Application_in_computational_mechanics?enrichId=rgreq-7ba3ccd9b6f798688d5342cc94e0bf49-XXX&enrichSource=Y292ZXJQYWdlOzM5NTU0MzUxNTtBUzoxMTQzMTI4MTYzNzIwOTExNUAxNzU4MDg4OTYwNzQ5&el=1_x_3)
  - VQLSのAの分解を不要にする手法
  - Aをvectorizationする(N xN → N²x1)

- ワークフロウ
  > 1. 問題設定: 解くべき線形システム Ku=f を定義
  > 2. 行列のベクトル化: 係数行列 K を行列分解する代わりに、**ベクトル化技術（Vectorization Technique）**を用いて K をベクトル vec(K) に変換
  >    ベクトル化演算子 vec(A) は、行列 A の列を順に積み重ねて列ベクトルを生成
  > 3. 量子状態の準備: 古典計算機上で、ベクトル化された行列 vec(K) を正規化し、量子状態 ∣vec(K)⟩ を生成するための準備
  >    右辺ベクトル f も同様に正規化され、量子状態 ∣f⟩ として準備
  >    この ∣vec(K)⟩ の次元は N^2×1 だが、量子振幅エンコーディングにより 2nq 量子ビットのみでエンコード可能 (nq=log_2N)
  > 4. アンザッツの実行: パラメータ θ を持つ変分量子回路 V(θ)（アンザッツ）を初期状態 ∣0⟩⊗nq に作用させ、試行解の状態 ∣u(θ)⟩=V(θ)∣0⟩⊗nq を生成
  > 5. コスト関数の計算: ∣u(θ)⟩ の正確さは、コスト関数 CG(θ)（大域的コスト関数）または CL(θ)（局所的コスト関数）を用いて評価
  >    DF-VQLSでは、これらのコスト関数の計算にスワップテスト（Swap Test）量子アルゴリズムを適用
  >    ◦ 内積への変換: ベクトル化技術により、コスト関数の分子や分母の項（例: ⟨f∣K∣u(θ)⟩ や ⟨u(θ)∣KTK∣u(θ)⟩）は、 ∣⟨ψ1∣ψ2⟩∣2 の形式を持つベクトルの内積の計算に変換される
  >    ◦ スワップテストの利用: スワップテスト回路は、2つの量子状態 ∣ψ1⟩ と ∣ψ2⟩ の内積の二乗 ∣⟨ψ1∣ψ2⟩∣2 を、アンシラ量子ビットの測定確率 P(0) から得られる
  > 6. パラメーター更新: 量子コンピューターから得られたコスト関数の値 C(θ) は、古典コンピューター上の最適化アルゴリズム（例: BFGS、COBYLA）に送られる
  > 7. 最小化: 古典最適化器は、コスト関数を最小化するようにアンザッツパラメーター θ を反復的に更新。収束条件を満たし、C(θ)→0 となる最適なパラメーター θ∗ が得られるまで
  > 8. 量子状態の読み出し: 最適化されたパラメーター θ∗ を用いて、最終的な量子状態 ∣u(θ∗)⟩ がアンザッツから読み出す
  > 9. スケール決定と最終解: 量子状態 ∣u(θ∗)⟩ は正規化されたベクトルであるため、元の真の解 u のスケールを回復するために、古典的にスケールファクター s を決定（例えば、二乗誤差ノルム ∣∣f−sK∣u(θ∗)⟩∣∣ を最小化する最小二乗フィッティングを用いる）
  > 10. 最終解の取得: 最終的な量子解は u=s∣u(θ∗)⟩ として得る

## つづきタスク

- 量子振幅エンコーディング(量子状態準備)のQiskit実装(initialize)の調査
- スワップテストの実装、コスト関数計算の実装を行う
- COBYLAを用いて動確

# 1104進捗 - DF-VQLS実装完了

## 実装ファイル

- `dfvqsl.py`: 8×8システム用のフル実装 (583行)
- `dfvqls_4x4.py`: 4×4システム用の高速テスト版（statevectorモード）

## 実装の詳細

### アーキテクチャ

**4×4システム (dfvqls_4x4.py)**:

- 分子回路: 9量子ビット (1 ancilla + 4 for |vec(K)⟩ + 4 for |u⟩⊗|f⟩)
- 分母回路: 13量子ビット (1 ancilla + 6 for |u⟩⊗|vec(K^T)⟩ + 6 for |vec(K)⟩⊗|u⟩)
- アンザッツ: Hardware-efficient (RY + CZ)
- シミュレーション: Statevector（完全精度、ショットノイズなし）

**8×8システム (dfvqsl.py)**:

- 分子回路: 13量子ビット
- 分母回路: 19量子ビット

### コスト関数の実装

論文の式:

```
CG(θ) = 1 - |⟨f|K|u(θ)⟩|² / ⟨u(θ)|K^T K|u(θ)⟩
```

スワップテストの関係式:

```
P(0) = (1 + |⟨ψ₁|ψ₂⟩|²) / 2
∴ |⟨ψ₁|ψ₂⟩|² = 2P(0) - 1
```

- 分子: `||K||² × |⟨vec(K)|u,f⟩|²` （二乗オーバーラップ使用 ✓）
- 分母: `||K||² × √(|⟨u⊗vec(K^T)|vec(K)⊗u⟩|²)` （平方根が必要！）

スワップテストは `|⟨ψ₁|ψ₂⟩|²` を返すが、分母は内積値そのもの `⟨u|K^T K|u⟩` が必要なため、平方根を取る必要がある。

## テスト結果

### 4×4三重対角システム

```python
K = [[2, -1,  0,  0],
     [-1, 2, -1,  0],
     [ 0, -1, 2, -1],
     [ 0,  0, -1, 2]]
f = [1, 0, 0, 1]
```

**結果**:

```
古典解: [1.0, 1.0, 1.0, 1.0]
DF-VQLS解: [0.9999, 0.9998, 0.9998, 0.9999]

最終コスト: 0.000000
相対誤差: 0.015%  ✓ Excellent!
反復回数: 138/200
レイヤー数: 3
```

## 実装の制約

### 1. 量子状態準備問題（最重要課題）

#### 振幅エンコーディングについて

DF-VQLSは**振幅エンコーディング** (amplitude encoding) を使用して古典データを量子状態に符号化:

```python
# 古典ベクトル vec(K) = [k₀, k₁, ..., k_{N²-1}] を量子振幅として符号化
|vec(K)⟩ = Σᵢ kᵢ|i⟩

# Qiskit実装
qc.initialize(vec_K_normalized, qubits)
```

#### シミュレータ vs 実量子デバイスの違い

| 環境           | `initialize()` の挙動              | 計算量        | 実用性               |
| -------------- | ---------------------------------- | ------------- | -------------------- |
| シミュレータ   | 状態ベクトルを直接メモリに書き込み | O(1)          | 高速・実用的         |
| 実量子デバイス | 量子ゲートに自動コンパイル         | O(2^n) ゲート | 指数的増大で非実用的 |

**本実装について**:

- `dfvqls.py`, `dfvqls_4x4.py` はシミュレータ向けに設計
- `initialize()` の使用は量子アルゴリズム研究の標準的手法
- アルゴリズムの正当性検証に適切

#### 実量子デバイスでの実行課題

**必要リソース (N×N行列)**:

- システム量子ビット: 6⌈log₂ N⌉ + 1 (問題なし)
- 状態準備用: 各 |vec(K)⟩ に 4log₂(N) 量子ビット

**具体例**:

- 8×8行列: 64次元 → 6量子ビット → O(64) ゲート以上
- N=1024 (論文より):
  - システム量子ビット: 20
  - 回路深さ: ~17
  - **必要な補助量子ビット: 約240万** (完全に非実用的)

### 2. 論文の評価

論文 Remark 2 より:

> "Efficient state preparation is a necessary requirement... This requirement inherits the well-known quantum input problem, a fundamental challenge."

> "The practical realization of DF-VQLS relies on future breakthroughs in state preparation and quantum hardware."

### 3. DF-VQLSが実用的となる問題クラス

以下の場合に実用性がある可能性:

**均一ベクトル**: `|+++⟩` 状態などはアダマールゲートで O(n) 準備可能

**構造的疎行列**: 非零要素が少数 (k << N) の場合、O(k log N) で準備可能

**特殊構造行列**: 巡回行列、Toeplitz行列など、効率的な量子回路が存在する可能性

### 4. オリジナルVQLS vs DF-VQLSのトレードオフ

**オリジナルVQLS**:

- 行列分解が必要: K = Σc_i U_i (O(N³) 前処理)
- O(k²) 回の回路実行
- ユニタリU_iは量子ゲートとして実装 (状態準備不要)

**DF-VQLS**:

- 行列分解不要
- 2回のみの回路実行
- N² 個の値を量子状態として符号化する必要

複雑性が「行列分解」から「量子状態準備」にシフト。

### 5. 本プロジェクトの位置づけ

**現在のスコープ**:

- シミュレータでのアルゴリズム実装・検証
- 数学的正当性の確認
- 小規模問題 (4×4, 8×8) でのテスト

**今後の方向性**:

- 問題特化型の状態準備回路の設計
- 近似的状態準備手法の検討
- 実量子デバイスでの小規模実証

---

# 反復追跡機能 (Iteration Tracking) - 2025-11-07追加

## 概要

DF-VQLSソルバーに**反復追跡機能**を実装しました。これにより、最適化の各反復における:
- パラメータ値 `θ`
- コスト関数値
- 解ベクトル（パラメータから再構築可能）

を記録・解析できるようになりました。

## 使用方法

### 基本的な使い方

```python
from src.linear_solvers import DFVQLSSolver
import numpy as np

# ソルバー作成
solver = DFVQLSSolver(
    matrix_size=4,
    num_layers=2,
    max_iter=100
)

# 反復追跡を有効にして解く
A = np.array([[2, -1, 0, 0],
              [-1, 2, -1, 0],
              [0, -1, 2, -1],
              [0, 0, -1, 2]])
b = np.array([1, 0, 0, 1])

x_solution, metadata, circuits = solver.solve(A, b, track_iterations=True)

# 反復履歴にアクセス
history = metadata['iteration_history']
for entry in history:
    print(f"Iter {entry['iteration']}: cost={entry['cost']:.6f}")
```

### 任意の反復での解を再構築

```python
# 50回目の反復での解を取得
params_at_50 = history[49]['params']  # 0-indexed
x_at_50 = solver.get_solution_at_params(params_at_50, A, b)

# 誤差計算
classical_sol = np.linalg.solve(A, b)
error_at_50 = np.linalg.norm(x_at_50 - classical_sol) / np.linalg.norm(classical_sol)
print(f"Error at iteration 50: {error_at_50:.6f}")
```

### 完全な収束曲線のプロット

```python
import matplotlib.pyplot as plt

# 全反復での誤差を計算
errors = []
classical_sol = np.linalg.solve(A, b)

for entry in history:
    x_iter = solver.get_solution_at_params(entry['params'], A, b)
    error = np.linalg.norm(x_iter - classical_sol) / np.linalg.norm(classical_sol)
    errors.append(error)

# プロット
plt.figure(figsize=(10, 6))
plt.semilogy(errors)  # 対数スケール
plt.xlabel('Iteration')
plt.ylabel('Relative Error')
plt.title('DF-VQLS Convergence')
plt.grid(True)
plt.show()
```

## 性能最適化: `compare.py`での活用

従来の`compare.py`は異なる`max_iter`値で**5回の独立した最適化**を実行していましたが、反復追跡により**1回の実行で全チェックポイントのデータ**を取得できるようになりました。

**改善前**:
```python
# 5回の独立した実行 (合計1500反復)
for max_iter in [100, 200, 300, 400, 500]:
    solver = DFVQLSSolver(max_iter=max_iter)
    x, metadata, circuits = solver.solve(A, b)
    # 結果を記録
```

**改善後**:
```python
# 1回の実行で全チェックポイントを取得 (合計500反復)
solver = DFVQLSSolver(max_iter=500)
x, metadata, circuits = solver.solve(A, b, track_iterations=True)

for checkpoint in [100, 200, 300, 400, 500]:
    params = metadata['iteration_history'][checkpoint-1]['params']
    x_checkpoint = solver.get_solution_at_params(params, A, b)
    # 各チェックポイントでの結果を記録
```

**性能向上**: 約**3倍高速化** (1500反復 → 500反復)

## 重要な制約: シミュレーション専用機能

⚠️ **この機能は statevector シミュレーション専用です**

### なぜシミュレーション専用なのか

`get_solution_at_params()`は以下のプロセスで解を再構築します:

1. パラメータ`θ`でアンザッツ回路を構築
2. **statevector simulation**を実行して`|u(θ)⟩`を取得
3. 完全な量子状態ベクトルから解を抽出
4. スケーリングを適用

### 実量子デバイスでの課題

実量子コンピュータでは:

| 項目 | シミュレータ | 実量子デバイス |
|------|--------------|----------------|
| 状態アクセス | 直接アクセス可能 | **不可能** (測定のみ) |
| 1反復あたりのコスト | O(1) | **O(shots)** (数千〜数万ショット必要) |
| 全反復での再構築 | 高速 (ms) | **非現実的** (数時間〜数日) |
| ノイズ | なし | **測定誤差、ゲート誤差** |

**具体例**:
- 100反復 × 1000ショット/反復 = 100,000回の回路実行
- 実量子デバイスでは非常に高コスト
- キューイング時間を含めると数時間〜数日

### 実量子デバイスでの代替アプローチ

実量子デバイスで収束を追跡するには:

1. **コスト関数値のみ記録** (既に実行済み)
   - 各反復でコスト値は計算済み
   - 解の再構築は不要

2. **選択的チェックポイント**
   - 最終解のみ、または数個のチェックポイントのみ測定
   - 全反復ではなく重要な点のみ

3. **測定ベース手法**
   - Tomography（完全な状態再構築、O(4^n)測定）
   - Shadow tomography（効率的な近似、O(n)測定）

## 活用シナリオ

### ✅ 適切な使用例

- **アルゴリズム研究**: 収束特性の解析
- **ベンチマーク**: 異なる手法の比較
- **デバッグ**: 最適化の振る舞いの理解
- **教育**: VQAの学習・可視化

### ❌ 不適切な使用例

- 実量子デバイスでの全反復追跡（コストが高すぎる）
- 大規模システム（状態ベクトルのメモリ制約）
- 本番環境での使用（シミュレーション専用）

## メモリ使用量

反復履歴のメモリオーバーヘッド（目安）:

- 2×2システム、500反復: ~50KB
- 4×4システム、500反復: ~100KB
- 8×8システム、500反復: ~200KB

アンザッツパラメータとコスト値のみ保存するため、メモリオーバーヘッドは最小限です。

## まとめ

反復追跡機能は**シミュレーション研究のための強力なツール**ですが、実量子デバイスでは使用できません。この制約は量子状態の測定原理（状態の読み出しは測定を通じてのみ可能）に由来する根本的なものです。

実量子デバイスでの実験では:
- コスト関数の収束曲線のみを追跡
- 最終解の精度評価に焦点を当てる
- 反復ごとの詳細解析は避ける
