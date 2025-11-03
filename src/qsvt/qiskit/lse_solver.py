import numpy as np
from inverse_matrix import InverseMatrix
from qiskit import QuantumCircuit
from qiskit_aer import Aer

SHOTS = 8192
# NOTE: ノイズなし → statevector_simulator。ノイズ付き → qasm_simulatorにしてnoise_modelを指定。
SIMULATOR = 'statevector_simulator'


class LSESolver:
    """
    連立一次方程式 Ax = b を解くためのクラス
    QSVTと古典的な方法の両方をサポート
    """

    def __init__(self, A, b, poly_degree: int, kappa: float):
        self.A = np.asarray(A, dtype=complex)
        self.b = np.asarray(b, dtype=complex)
        self.n, self.m = self.A.shape
        self.qubits = self.n+2
        self.inverse_matrix_solver = InverseMatrix(self.A, poly_degree=poly_degree, kappa=kappa)

    def solve_lse_classical(self):
        """
        古典的な方法で連立一次方程式 Ax = b を解く
        """
        A = np.asarray(self.A, dtype=complex)
        b = np.asarray(self.b, dtype=complex)

        A_inv = np.linalg.inv(A)
        x_solution = A_inv @ b

        residual_norm = np.linalg.norm(A @ x_solution - b)

        return {
            'solution': x_solution,
            'residual_norm': residual_norm
        }

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
        unitary, qsvt_circuit, encoding_wires, _ = self.inverse_matrix_solver.compute_matrix_inverse_qsvt()

        # TODO: 固定値にしているので、変更する。
        sys_wires = [1]
        # anc_wires = meta.get("ancilla_wires", [1])  # 必要なら

        # 3) 合成用の土台回路は “QSVT回路と同じ本数” にする
        combined_circuit = QuantumCircuit(qsvt_circuit.num_qubits)

        # 4) |b> を “系” の量子ビットにだけ初期化
        combined_circuit.initialize(b_normalized, sys_wires)

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

        print("bnorm:", b_norm)

        # Step 7: 測定結果から解ベクトルを再構成
        x_solution = self._reconstruct_from_counts(counts)

        s = self.inverse_matrix_solver.s
        max_singular_value = self.inverse_matrix_solver.max_singular_value
        # スケールを調整
        x_solution = x_solution*b_norm / (s*max_singular_value)

        return x_solution

    def _reconstruct_from_counts(self, counts):
        """
        現在の配置（|anc>|b> の順＝qubit0=anc、qubit1=system）に対応。
        anc=0 のときの条件付き確率から振幅の絶対値を復元。
        """
        # anc=0のカウントだけ抜く
        # NOTE: Qiskitの仕様上最初のビットが最右に記載される。
        post_counts = {bits: c for bits, c in counts.items() if bits[-1] == '0'}
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

    def compare_lse_solutions(self, classical_result, qsvt_result):
        """
        古典解とQSVT解を比較
        """
        print("="*40)
        print("LINEAR SYSTEM SOLUTION COMPARISON")
        print("="*40)

        print(f"Classical: {np.round(classical_result['solution'], 4)}")
        print(f"QSVT:      {np.round(qsvt_result, 4)}")

    def test_lse_solvers(self):
        """
        両方の方法をテストする関数
        """
        classical_result = self.solve_lse_classical()
        qsvt_result = self.solve_linear_system_quantum()

        self.compare_lse_solutions(classical_result, qsvt_result)

        return classical_result, qsvt_result

    def main(self):
        """
        メイン関数：QSVT逆行列計算のテスト
        """
        print("QSVT Matrix Inverse Computation Test")
        print("=" * 50)
        print(f"Test matrix A:\n{np.round(A.real, 4)}")

        # QSVT逆行列の計算
        qsvt_inv, qsvt_circuit, encoding_wires, poly_coeffs = self.inverse_matrix_solver.compute_matrix_inverse_qsvt()

        print(f"\nQSVT Inverse Matrix:\n{np.round(qsvt_inv.real, 4)}")

        # 古典逆行列の計算
        classical_inv = self.inverse_matrix_solver.compute_classical_inverse()
        print(f"\nClassical Inverse Matrix:\n{np.round(classical_inv.real, 4)}")

        self.test_lse_solvers()


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

    lse_solver = LSESolver(A=A, b=b, poly_degree=100, kappa=20.0)
    lse_solver.main()
