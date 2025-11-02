# 参照資料

[Qiita](https://qiita.com/Ayumu_walker/items/6b1131c6be64a8c5ad37)
[arxiv](https://arxiv.org/abs/1909.05820)
[Github](https://github.com/Qiskit/textbook/blob/main/notebooks/ch-applications/vqls.ipynb)

# 既存コードからの変更点

- aer, assembleがdeprecatedになったため、qiskit_aerとAerSimulator.runへ変更

# タスク

- 理解していないこと
  - 何故アンザッツはその形を使用しているか
  - 一般の問題に対して行列Aをユニタリ行列の線形結合にどう分解するか - これは発表に必須なので、要調査
- サブ目標
  - VQLSの回転パラメタを、はじめはクリフォードにすることでゲート分解が高速にできるか試したい
    - ソースは[これ](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.062413)のはず
