import numpy as np
from qiskit_aer import Aer
from qsvt import qsvt, transform_angles
import pyqsp
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from pyqsp.poly import PolyOneOverX


def compute_matrix_inverse_qsvt(A, angles_qsvt):
    """
    QSVTを使って行列の逆行列を計算
    Aは正規化されていることを前提とする。

    Returns:
        tuple: (QSVT逆行列, QSVT回路, 多項式係数)
    """
    # 量子ビット数を計算
    n, m = A.shape
    required_qubits = int(np.ceil(np.log2(n)))
    encoding_wires = list(range(required_qubits))

    qsvt_circuit, phase_matrices = qsvt(A, angles_qsvt, encoding_wires)
    full_unitary = get_full_unitary(qsvt_circuit)

    return full_unitary, qsvt_circuit, encoding_wires, phase_matrices


def get_full_unitary(qsvt_circuit):
    unitary_backend = Aer.get_backend("unitary_simulator")
    unitary_job = unitary_backend.run(qsvt_circuit)
    unitary_result = unitary_job.result()
    full_unitary = unitary_result.get_unitary(qsvt_circuit)
    return full_unitary


def compute_classical_inverse(A):
    return np.linalg.inv(A)


def confirm_inverse_matrix_in_unitary(A, s, frobenius_norm, qsvt_circuit):
    """
    QSVT回路のユニタリ行列に逆行列が部分行列として存在することを確認
    """
    unitary_backend = Aer.get_backend("unitary_simulator")
    unitary_job = unitary_backend.run(qsvt_circuit)
    unitary_result = unitary_job.result()
    full_unitary = unitary_result.get_unitary(qsvt_circuit)

    n, m = A.shape
    P_A = full_unitary[:n, :m] / s / frobenius_norm
    return P_A, full_unitary


def generate_angles_qsvt_and_scale(kappa: float) -> tuple[np.ndarray, float]:
    pcoefs, s = pyqsp.poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True)
    phi_pyqsp = pyqsp.angle_sequence.QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", tolerance=0.00001)
    phi_qsvt = transform_angles(phi_pyqsp, "QSP", "QSVT")
    return phi_qsvt, s
