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

1. **量子状態準備問題**: N×N行列では N² 次元の vec(K) が必要で、2log₂(N²) = 4log₂(N) 量子ビットが必要
   - 8×8 → 64次元 → 6量子ビット
   - しかし `initialize()` は O(2^n) ゲートを生成するため、実用上は困難

2. **論文の言及**: Remark 2で「量子入力問題」として認識されており、実用にはO(10⁶)の補助量子ビットが必要
