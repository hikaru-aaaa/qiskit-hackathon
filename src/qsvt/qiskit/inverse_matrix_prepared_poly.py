import numpy as np
from qiskit_aer import Aer
from qsvt_qiskit_prepared_poly import qsvt
from param_store import load_angles


class InverseMatrix:
    def __init__(self, A, kappa=50.0, epsilon=0.01):
        self.A = np.asarray(A, dtype=complex)
        self.kappa = kappa
        self.epsilon = epsilon
        self.frobenius_norm = None
        self.s = 1.0 / self.kappa

    def compute_matrix_inverse_qsvt(self, angles_qsvt=None):
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

        print("A_normalized:\n", np.round(A_normalized, 4))

        # 量子ビット数を計算
        n, m = A_array.shape
        required_qubits = int(np.ceil(np.log2(n)))
        encoding_wires = list(range(required_qubits))

        if angles_qsvt is None:
            angles_qsvt = load_angles(self.epsilon, self.kappa)
        # print("angles_qsvt", angles_qsvt)
        # angles_qsvt = [x for x in angles_qsvt]

        qsvt_circuit, phase_matrices = qsvt(A_normalized, angles_qsvt, encoding_wires)

        scaled_inverse_matrix, full_unitary = self.confirm_inverse_matrix_in_unitary(
            qsvt_circuit
        )
        print(f"scaled_inverse_matrix:\n{np.round(scaled_inverse_matrix, 4)}")
        print(f"full_unitary:\n{np.round(full_unitary, 4)}")

        return full_unitary, qsvt_circuit, encoding_wires, phase_matrices

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
