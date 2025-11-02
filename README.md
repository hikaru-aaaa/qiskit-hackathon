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
  - b = U|0>を満すようなHの探し方(今回はH⊗3だが一般には？)
- サブ目標
  - VQLSの回転パラメタを、はじめはクリフォードにすることでゲート分解が高速にできるか試したい
    - ソースは[これ](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.062413)のはず
