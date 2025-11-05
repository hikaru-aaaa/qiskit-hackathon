"""
Inverse Matrix Computation using QSVT

QSVTを使って逆行列を計算するモジュール
"""

import numpy as np
from typing import Tuple
from qiskit_aer import Aer

# 相対インポートでqsvtモジュールをインポート
import sys
from pathlib import Path

# プロジェクトルートを取得
project_root = Path(__file__).parent.parent.parent.parent
qsvt_path = Path(__file__).parent.parent

# パスを追加
if str(qsvt_path) not in sys.path:
    sys.path.insert(0, str(qsvt_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # 最新のqsvtブランチの実装を使用
    from qsvt import qsvt, transform_angles
    import pyqsp
    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    from pyqsp.poly import PolyOneOverX
    USE_PYQSP = True
except ImportError:
    try:
        # フォールバック: 古い実装（qiskit配下）
        qiskit_path = Path(__file__).parent.parent / 'qiskit'
        if str(qiskit_path) not in sys.path:
            sys.path.insert(0, str(qiskit_path))
        from poly_approximation import fit_invx_poly
        from qsvt_qiskit_pennylane import qsvt
        USE_PYQSP = False
    except ImportError:
        raise ImportError(
            f"QSVT modules not found. "
            f"Tried paths: {qsvt_path}, {qiskit_path if 'qiskit_path' in locals() else 'N/A'}"
        )


class InverseComputer:
    """
    QSVTを使って逆行列を計算するクラス
    
    Generalized DF-VQLSと同様のモジュラー設計を採用。
    """
    
    def __init__(self, poly_degree: int = 100, kappa: float = 20.0):
        """
        初期化
        
        Args:
            poly_degree: 多項式近似の次数
            kappa: 条件数（condition number）の推定値
        """
        self.poly_degree = poly_degree
        self.kappa = kappa
        self.frobenius_norm = None
        self.s = None
        self.poly_coeffs = None
    
    def _generate_angles_qsvt_and_scale(self) -> Tuple[np.ndarray, float]:
        """
        最新のqsvtブランチの実装に合わせて、pyqspを使用して角度を生成
        
        Returns:
            Tuple of (QSVT角度, スケール係数s)
        """
        if USE_PYQSP:
            # 最新の実装: pyqspを使用
            pcoefs, s = pyqsp.poly.PolyOneOverX().generate(
                self.kappa, 
                return_coef=True, 
                ensure_bounded=True, 
                return_scale=True
            )
            phi_pyqsp = pyqsp.angle_sequence.QuantumSignalProcessingPhases(
                pcoefs, 
                signal_operator="Wx", 
                tolerance=0.00001
            )
            phi_qsvt = transform_angles(phi_pyqsp, "QSP", "QSVT")
            return phi_qsvt, s
        else:
            # フォールバック: 古い実装
            delta = 1.0 / self.kappa
            s = 0.99 * delta  # |P(x)| ≤ 1 を安全に満たすため少し小さめに
            
            raw_coeffs = fit_invx_poly(delta=delta, degree=self.poly_degree)[5]
            
            # スケールと奇関数性の強制
            coeffs = []
            for i, c in enumerate(raw_coeffs):
                coeffs.append(0.0 if i % 2 == 0 else s * c)
            
            # 注意: 古い実装では係数を返すが、最新の実装では角度を返す
            # 互換性のため、係数をそのまま返す（後で角度に変換する必要がある）
            return np.array(coeffs), s
    
    def _build_inv_poly_coeffs(self, s: float = None, safety: float = 0.99) -> Tuple[list, float]:
        """
        1/x の多項式近似係数を QSVT 用にスケールして作る。
        P(x) ≈ s/x （x ∈ [δ,1], δ=1/κ）かつ |P(x)| ≤ 1 を満たすよう調整。
        偶数次数は 0（奇関数化）。
        
        Args:
            s: スケーリング係数（Noneの場合は自動計算）
            safety: 安全性係数
            
        Returns:
            Tuple of (多項式係数, s)
        """
        # 最新の実装では使用しない（後方互換性のため残す）
        delta = 1.0 / self.kappa
        if s is None:
            s = safety * delta
        
        if USE_PYQSP:
            # pyqspを使用する場合は_generate_angles_qsvt_and_scaleを使用
            angles, s = self._generate_angles_qsvt_and_scale()
            # 角度から係数への変換は複雑なので、角度をそのまま返す
            return angles, s
        else:
            # フォールバック: 古い実装
            raw_coeffs = fit_invx_poly(delta=delta, degree=self.poly_degree)[5]
            
            coeffs = []
            for i, c in enumerate(raw_coeffs):
                coeffs.append(0.0 if i % 2 == 0 else s * c)
            
            return coeffs, s
    
    def compute_matrix_inverse_qsvt(
        self, 
        A: np.ndarray
    ) -> Tuple[np.ndarray, 'QuantumCircuit', list, list]:
        """
        QSVTを使って行列の逆行列を計算
        
        Args:
            A: 入力行列
            
        Returns:
            Tuple of (完全ユニタリ, QSVT回路, encoding_wires, 多項式係数)
        """
        A_array = np.asarray(A, dtype=complex)
        n, m = A_array.shape
        
        # 行列の正規化
        # 最大特異値をフロベニウスノルムで抑える
        self.frobenius_norm = np.linalg.norm(A_array, ord="fro")
        A_normalized = A_array / self.frobenius_norm
        
        # 最新の実装: pyqspを使用して角度を生成
        if USE_PYQSP:
            angles_qsvt, self.s = self._generate_angles_qsvt_and_scale()
            # 量子ビット数を計算
            required_qubits = int(np.ceil(np.log2(n)))
            encoding_wires = list(range(required_qubits))
            
            # QSVT回路を構築（最新の実装では角度を使用）
            qsvt_circuit, phase_matrices = qsvt(A_normalized, angles_qsvt, encoding_wires)
            self.poly_coeffs = phase_matrices  # 後方互換性のため
        else:
            # フォールバック: 古い実装
            self.poly_coeffs, self.s = self._build_inv_poly_coeffs()
            required_qubits = int(np.ceil(np.log2(n)))
            encoding_wires = list(range(required_qubits))
            qsvt_circuit = qsvt(A_normalized, self.poly_coeffs, encoding_wires)
        
        # 完全ユニタリを取得（検証用）
        full_unitary = self._get_full_unitary(qsvt_circuit)
        
        return full_unitary, qsvt_circuit, encoding_wires, self.poly_coeffs
    
    def _get_full_unitary(self, qsvt_circuit) -> np.ndarray:
        """
        QSVT回路の完全ユニタリ行列を取得
        
        Args:
            qsvt_circuit: QSVT回路
            
        Returns:
            完全ユニタリ行列
        """
        unitary_backend = Aer.get_backend("unitary_simulator")
        unitary_job = unitary_backend.run(qsvt_circuit)
        unitary_result = unitary_job.result()
        full_unitary = unitary_result.get_unitary(qsvt_circuit)
        return np.asarray(full_unitary)
    
    def extract_inverse_matrix(self, full_unitary: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        完全ユニタリから逆行列を抽出
        
        Args:
            full_unitary: QSVT回路の完全ユニタリ
            A: 元の行列（サイズ情報用）
            
        Returns:
            抽出された逆行列
        """
        n, m = A.shape
        P_A = full_unitary[:n, :m] / self.s / self.frobenius_norm
        return P_A

