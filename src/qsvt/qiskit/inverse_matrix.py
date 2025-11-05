import numpy as np
from poly_approximation import fit_invx_poly
from qiskit_aer import Aer
from qsvt_qiskit_pennylane import qsvt


class InverseMatrix:
    def __init__(self, A, poly_degree, kappa=20.0):
        self.A = np.asarray(A, dtype=complex)
        self.poly_degree = poly_degree
        self.kappa = kappa
        self.frobenius_norm = None
        self.s = None
        self.poly_coeffs, self.s = self._build_inv_poly_coeffs()

    def _build_inv_poly_coeffs(self, s=None, safety=0.99):
        """
        1/x の多項式近似係数を QSVT 用にスケールして作る。
        P(x) ≈ s/x （x ∈ [δ,1], δ=1/κ）かつ |P(x)| ≤ 1 を満たすよう調整。
        偶数次数は 0（奇関数化）。
        """
        delta = 1.0 / self.kappa
        if self.s is None:
            self.s = safety * delta  # |P(x)| ≤ 1 を安全に満たすため少し小さめに

        # fit_invx_poly は 1/x 近似の係数（昇冪）を返す想定
        raw_coeffs = fit_invx_poly(delta=delta, degree=self.poly_degree)[5]

        # スケールと奇関数性の強制
        coeffs = []
        for i, c in enumerate(raw_coeffs):
            coeffs.append(0.0 if i % 2 == 0 else self.s * c)

        return coeffs, self.s

    def compute_matrix_inverse_qsvt(self):
        """
        QSVTを使って行列の逆行列を計算

        Returns:
            tuple: (QSVT逆行列, QSVT回路, 多項式係数)
        """
        A_array = np.asarray(self.A, dtype=complex)

        # 行列の正規化
        # 最大特異値をフロベニウスノルムで抑える
        self.frobenius_norm = np.linalg.norm(A_array, ord="fro")
        A_normalized = A_array / self.frobenius_norm

        # 多項式係数を生成
        normalized_poly_coeffs, self.s = self._build_inv_poly_coeffs()

        # 量子ビット数を計算
        n, m = A_array.shape
        required_qubits = int(np.ceil(np.log2(n)))
        encoding_wires = list(range(required_qubits))

        qsvt_circuit = qsvt(A_normalized, normalized_poly_coeffs, encoding_wires)
        # sv_sim = StatevectorSimulator()
        # transpiled_circuit = transpile(qsvt_circuit, sv_sim)
        # result = sv_sim.run(transpiled_circuit).result()
        # statevector_obj = result.get_statevector()
        # statevector = np.asarray(statevector_obj)

        scaled_inverse_matrix, full_unitary = self.confirm_inverse_matrix_in_unitary(
            qsvt_circuit
        )
        print(f"scaled_inverse_matrix:\n{np.round(scaled_inverse_matrix, 4)}")
        print(f"full_unitary:\n{np.round(full_unitary, 4)}")

        return full_unitary, qsvt_circuit, encoding_wires, normalized_poly_coeffs

    def compute_classical_inverse(self):
        return np.linalg.inv(self.A)

    def confirm_inverse_matrix_in_unitary(self, qsvt_circuit):
        """
        QSVT回路のユニタリ行列に逆行列が部分行列として存在することを確認
        """
        unitary_backend = Aer.get_backend("unitary_simulator")
        unitary_job = unitary_backend.run(qsvt_circuit)
        unitary_result = unitary_job.result()
        full_unitary = unitary_result.get_unitary(qsvt_circuit)

        n, m = self.A.shape
        P_A = full_unitary[:n, :m] / self.s / self.frobenius_norm
        return P_A, full_unitary
