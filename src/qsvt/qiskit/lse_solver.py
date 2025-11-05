import numpy as np
from inverse_matrix import compute_matrix_inverse_qsvt, generate_angles_qsvt_and_scale
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector

SHOTS = 8192
# NOTE: ノイズなし → statevector_simulator。ノイズ付き → qasm_simulatorにしてnoise_modelを指定。
SIMULATOR = "statevector_simulator"


class LSESolver:
    """
    連立一次方程式 Ax = b を解くためのクラス
    QSVTと古典的な方法の両方をサポート
    """

    def __init__(self, A, b):
        self.A = np.asarray(A, dtype=complex)
        self.b = np.asarray(b, dtype=complex)
        self.n, self.m = self.A.shape

        self.frobenius_norm = None
        self.A_normalized = None
        self.normalize_matrix()

    def normalize_matrix(self):
        # NOTE: 行列Aの特異値が1以下になるように正規化。
        self.frobenius_norm = np.linalg.norm(self.A, ord="fro")
        self.A_normalized = self.A / self.frobenius_norm

    def compute_kappa(self):
        # TODO: 後で実装。
        return 4

    def solve_lse_classical(self):
        """
        古典的な方法で連立一次方程式 Ax = b を解く
        """
        A_inv = np.linalg.inv(self.A)
        x_solution = A_inv @ self.b

        residual_norm = np.linalg.norm(self.A @ x_solution - self.b)

        return {"solution": x_solution, "residual_norm": residual_norm}

    def solve_linear_system_quantum(self):
        """
        実際の量子デバイスでの連立一次方程式の解法
        """
        # TODO: 符号の情報が抜け落ちるのでアダマール変換組み込む？
        # TODO: まだはじめのバージョンなので2x2限定なので、一般の正方行列で実装する必要あり。
        # 変更部分は補助ビットの数などだと思われる。
        # TODO: 逆行列まではある程度精度高くもとまるが、最終のxの値が間違っているので修正。

        # 1) b を正規化（複素数型にしておくと無難）
        b_norm = np.linalg.norm(self.b)
        b_normalized = (self.b / b_norm).astype(complex)

        # 2) QSVT回路を取得（戻り値は適宜合わせて）
        #    qsvt_circuit: ユニタリ全体）

        kappa = self.compute_kappa()
        angles_qsvt, s = generate_angles_qsvt_and_scale(kappa)

        full_unitary, qsvt_circuit, encoding_wires, _ = compute_matrix_inverse_qsvt(
            self.A_normalized,  angles_qsvt
        )
        print(f"full_unitary:\n{np.round(full_unitary, 4)}")
        print(f"scale: {s}")

        n_sys = int(np.ceil(np.log2(len(self.b))))
        sys_wires = list[int, ...](range(1, n_sys + 1))

        # 3) 合成用の土台回路は “QSVT回路と同じ本数” にする
        combined_circuit = QuantumCircuit(qsvt_circuit.num_qubits)

        # 4) |b> を “系” の量子ビットにだけ初期化
        combined_circuit.initialize(b_normalized, sys_wires)
        init_state = Statevector.from_instruction(combined_circuit)
        print("Initial statevector before QSVT:")
        print(np.round(init_state.data, 4))

        # 5) QSVT を合成（qubit 並びが一致していればOK）
        combined_circuit.compose(qsvt_circuit, inplace=True)

        # Step 5: 測定で解xを取得
        # システムレジスタのみ測定（アンシラは無視）
        combined_circuit.measure_all()

        # Step 6: 実行と後処理
        backend = Aer.get_backend(SIMULATOR)
        result = backend.run(combined_circuit, shots=SHOTS).result()
        counts = result.get_counts()
        print("Measurement results:")
        print(counts)

        # Step 7: 測定結果から解ベクトルを再構成
        x_solution = self._reconstruct_from_counts(counts)
        # スケールを調整
        x_solution = x_solution * b_norm / (s * self.frobenius_norm)

        return x_solution

    def _reconstruct_from_counts(self, counts):
        """
        現在の配置（|anc>|b> の順＝qubit0=anc、qubit1=system）に対応。
        anc=0 のときの条件付き確率から振幅の絶対値を復元。
        """
        # anc=0のカウントだけ抜く
        # NOTE: Qiskitの仕様上最初のビットが最右に記載される。
        post_counts = {bits: c for bits, c in counts.items() if bits[-1] == "0"}
        total_post = sum(post_counts.values())

        if total_post == 0:
            raise RuntimeError("anc=0 のデータがありません。")

        # systemの確率
        p_sys = [0.0, 0.0]
        for bits, c in post_counts.items():
            sys_bit = bits[0]  # 左側がsystem
            p_sys[int(sys_bit)] += c

        # 条件付き確率に正規化
        p_sys = [p / SHOTS for p in p_sys]

        # 振幅の大きさ
        solution = np.sqrt(p_sys)
        return solution

    def run_both_solutions(self):
        """
        古典解とQSVT解を比較
        """
        print("=" * 40)
        print("LINEAR SYSTEM SOLUTION COMPARISON")
        print("=" * 40)

        classical_result = self.solve_lse_classical()
        qsvt_result = self.solve_linear_system_quantum()

        print(f"Classical: {np.round(classical_result['solution'], 4)}")
        print(f"QSVT:      {np.round(qsvt_result, 4)}")


def qsvt_solve_linear_system(A, b):
    lse_solver = LSESolver(A=A, b=b)
    return lse_solver.solve_linear_system_quantum()


if __name__ == "__main__":
    A = np.array([3, 1, 1, 3]).reshape((2, 2))
    b = np.array([1, 2], dtype="complex")

    # NOTE: より大きな行列のテスト用に残しておく。
    # A = np.array(
    #     [
    #         [0.65713691, -0.05349524, 0.08024556, -0.07242864],
    #         [-0.05349524, 0.65713691, -0.07242864, 0.08024556],
    #         [0.08024556, -0.07242864, 0.65713691, -0.05349524],
    #         [-0.07242864, 0.08024556, -0.05349524, 0.65713691],
    #     ]
    # )

    # b = np.array([1, 2, 3, 4], dtype="complex")

    x_solution = qsvt_solve_linear_system(A, b)
    print(f"x_solution: {np.round(x_solution, 4)}")
