"""
QSVT Solver for Linear Systems

Quantum Singular Value Transformation (QSVT) を使用して
線形方程式系 Ax = b を解くソルバー
"""

import numpy as np
from typing import Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator

from .inverse_computer import InverseComputer


class QSVTSolver:
    """
    QSVT-based Linear System Solver
    
    Generalized DF-VQLSと同様のインターフェースを提供します。
    """
    
    def __init__(
        self,
        matrix_size: int,
        poly_degree: int = 100,
        kappa: float = 20.0,
        random_seed: int = None,
        verbose: bool = True,
        use_statevector: bool = True
    ):
        """
        初期化
        
        Args:
            matrix_size: 行列サイズ（2のべき乗のみ）
            poly_degree: 多項式近似の次数
            kappa: 条件数（condition number）の推定値
            random_seed: ランダムシード（未使用だが互換性のため）
            verbose: 詳細出力
            use_statevector: statevectorモードを使用（True）または測定モード（False）
        """
        self.matrix_size = matrix_size
        self.poly_degree = poly_degree
        self.kappa = kappa
        self.verbose = verbose
        self.use_statevector = use_statevector
        
        # InverseComputerを初期化
        self.inverse_computer = InverseComputer(
            poly_degree=poly_degree,
            kappa=kappa
        )
        
        # シミュレータを初期化
        if use_statevector:
            self.simulator = AerSimulator(method='statevector')
        else:
            self.simulator = AerSimulator(method='qasm')
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        線形方程式系 Ax = b をQSVTで解く
        
        Args:
            A: 係数行列 (N×N)
            b: 右辺ベクトル (N,)
            
        Returns:
            Tuple of (解ベクトル, メタデータ)
        """
        A = np.asarray(A, dtype=complex)
        b = np.asarray(b, dtype=complex)
        
        # 入力検証
        if A.shape != (self.matrix_size, self.matrix_size):
            raise ValueError(
                f"Matrix A must be {self.matrix_size}×{self.matrix_size}, "
                f"got {A.shape}"
            )
        
        if b.shape not in [(self.matrix_size,), (self.matrix_size, 1)]:
            raise ValueError(
                f"Vector b must be {self.matrix_size}-dimensional, "
                f"got {b.shape}"
            )
        
        b = b.flatten()  # 1Dに変換
        
        if self.verbose:
            print("=" * 70)
            print(f"QSVT for {self.matrix_size}×{self.matrix_size} System")
            print("=" * 70)
            print(f"Matrix size: {self.matrix_size}×{self.matrix_size}")
            print(f"Polynomial degree: {self.poly_degree}")
            print(f"Condition number (kappa): {self.kappa}")
            print(f"Mode: {'Statevector' if self.use_statevector else 'Measurement'}")
            print("=" * 70 + "\n")
        
        # bを正規化
        b_norm = np.linalg.norm(b)
        b_normalized = (b / b_norm).astype(complex)
        
        # QSVT回路を取得
        full_unitary, qsvt_circuit, encoding_wires, poly_coeffs = (
            self.inverse_computer.compute_matrix_inverse_qsvt(A)
        )
        
        # 最新のqsvtブランチの実装: Hadamardテスト（Overlap Test）を使用
        # これにより符号情報も含めて正確に解を抽出できる
        real_amplitudes = self._measure_real_amplitudes(
            qsvt_circuit, b_normalized, encoding_wires
        )
        
        # 解を再構成
        x_quantum = self._reconstruct_from_real_amplitudes(real_amplitudes)
        
        # スケール調整
        s = self.inverse_computer.s
        frobenius_norm = self.inverse_computer.frobenius_norm
        x_quantum = x_quantum * b_norm / (s * frobenius_norm)
        
        # メタデータ
        metadata = {
            'method': 'QSVT',
            'poly_degree': self.poly_degree,
            'kappa': self.kappa,
            'frobenius_norm': self.inverse_computer.frobenius_norm,
            's': self.inverse_computer.s
        }
        
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"QSVT solution computed")
            print(f"Frobenius norm: {self.inverse_computer.frobenius_norm:.6f}")
            print(f"Scaling factor s: {self.inverse_computer.s:.6e}")
            print("=" * 70)
        
        return x_quantum, metadata
    
    def _solve_with_statevector(
        self,
        qsvt_circuit: QuantumCircuit,
        b_normalized: np.ndarray,
        encoding_wires: list,
        b_norm: float,
        A: np.ndarray
    ) -> np.ndarray:
        """
        Statevectorモードで解を計算（より正確）
        
        Args:
            qsvt_circuit: QSVT回路
            b_normalized: 正規化された右辺ベクトル
            encoding_wires: エンコーディング用の量子ビット
            b_norm: 右辺ベクトルのノルム
            A: 元の行列
            
        Returns:
            解ベクトル
        """
        from qiskit import transpile
        
        # システム量子ビットのインデックス（encoding_wiresの最初）
        # 注: 現在の実装は2×2限定なので、一般化が必要
        sys_wires = encoding_wires[0:1] if len(encoding_wires) == 1 else encoding_wires
        
        # 合成回路を作成
        combined_circuit = QuantumCircuit(qsvt_circuit.num_qubits)
        
        # |b>を初期化（システム量子ビットにのみ）
        # 注意: 現在の実装は2×2限定のため、一般化が必要
        if self.matrix_size == 2:
            combined_circuit.initialize(b_normalized, [1])
        else:
            # より大きな行列の場合の一般化が必要
            raise NotImplementedError(
                f"Statevector mode for {self.matrix_size}×{self.matrix_size} "
                "matrices is not yet implemented. Use measurement mode."
            )
        
        # QSVT回路を合成
        combined_circuit.compose(qsvt_circuit, inplace=True)
        combined_circuit.save_statevector()
        
        # 実行
        transpiled = transpile(combined_circuit, self.simulator)
        result = self.simulator.run(transpiled).result()
        statevector = np.asarray(result.get_statevector(combined_circuit))
        
        # 状態ベクトルから解を抽出
        # 注意: これは一般化が必要（現在は2×2限定）
        if self.matrix_size == 2:
            # 2×2の場合: アンシラ=0の状態から解を抽出
            x_quantum = np.zeros(2, dtype=complex)
            for i in range(2):
                # アンシラ=0, システム=iの状態
                idx = i * 2  # 簡略化（実際の実装は回路構造に依存）
                if idx < len(statevector):
                    x_quantum[i] = statevector[idx]
            
            # スケール調整
            s = self.inverse_computer.s
            frobenius_norm = self.inverse_computer.frobenius_norm
            x_quantum = x_quantum * b_norm / (s * frobenius_norm)
            
            return np.real(x_quantum)
        else:
            raise NotImplementedError(
                f"Statevector mode for {self.matrix_size}×{self.matrix_size} "
                "matrices is not yet implemented."
            )
    
    def _solve_with_measurement(
        self,
        qsvt_circuit: QuantumCircuit,
        b_normalized: np.ndarray,
        encoding_wires: list,
        b_norm: float,
        A: np.ndarray
    ) -> np.ndarray:
        """
        測定モードで解を計算（実デバイス向け）
        
        Args:
            qsvt_circuit: QSVT回路
            b_normalized: 正規化された右辺ベクトル
            encoding_wires: エンコーディング用の量子ビット
            b_norm: 右辺ベクトルのノルム
            A: 元の行列
            
        Returns:
            解ベクトル
        """
        from qiskit import transpile
        
        SHOTS = 8192
        
        # システム量子ビットのインデックス
        # 注意: 現在の実装は2×2限定
        if self.matrix_size == 2:
            sys_wires = [1]
        else:
            raise NotImplementedError(
                f"Measurement mode for {self.matrix_size}×{self.matrix_size} "
                "matrices is not yet implemented."
            )
        
        # 合成回路を作成
        combined_circuit = QuantumCircuit(qsvt_circuit.num_qubits)
        
        # |b>を初期化
        combined_circuit.initialize(b_normalized, sys_wires)
        
        # QSVT回路を合成
        combined_circuit.compose(qsvt_circuit, inplace=True)
        
        # 測定
        combined_circuit.measure_all()
        
        # 実行
        transpiled = transpile(combined_circuit, self.simulator)
        result = self.simulator.run(transpiled, shots=SHOTS).result()
        counts = result.get_counts()
        
        if self.verbose:
            print("Measurement results:")
            print(counts)
        
        # 測定結果から解を再構成
        x_quantum = self._reconstruct_from_counts(counts, SHOTS)
        
        # スケール調整
        s = self.inverse_computer.s
        frobenius_norm = self.inverse_computer.frobenius_norm
        x_quantum = x_quantum * b_norm / (s * frobenius_norm)
        
        return x_quantum
    
    def _measure_real_amplitudes(
        self,
        qsvt_circuit: QuantumCircuit,
        b_normalized: np.ndarray,
        encoding_wires: list
    ) -> Dict[str, float]:
        """
        Hadamardテスト（Overlap Test）を用いて全基底状態の確率振幅の実部を測定
        
        最新のqsvtブランチの実装を正確に再現。
        QSVT適用後の状態 |ψ⟩ = Σᵢ αᵢ|i⟩ の各振幅の実部 Re(αᵢ) を測定
        
        Args:
            qsvt_circuit: QSVT回路
            b_normalized: 正規化された右辺ベクトル
            encoding_wires: エンコーディング用の量子ビット
            
        Returns:
            全基底状態の実振幅の辞書
        """
        from qiskit_aer import Aer
        
        SHOTS = 8192
        SIMULATOR = "statevector_simulator"
        
        # システム情報の定義
        n_sys = int(np.ceil(np.log2(len(b_normalized))))
        sys_wires = list(range(n_sys))  # |b⟩ を準備するワイヤ
        n_qubits = qsvt_circuit.num_qubits  # システム全体のqubit数
        sys_qubits = list(range(n_qubits))  # QSVT回路が作用する全ワイヤ
        backend = Aer.get_backend(SIMULATOR)
        
        # 全基底状態の数（2^n_qubits）
        n_total_states = 2**n_qubits
        
        real_amplitudes = {}
        
        if self.verbose:
            print("=" * 60)
            print("Hadamardテストによる全基底状態の実部測定")
            print(
                f"総qubit数: {n_qubits} (うち|b⟩初期化: {n_sys}), 基底状態数: {n_total_states}"
            )
            print("=" * 60)
        
        # 各基底状態に対してHadamardテストを実行
        for basis_idx in range(n_total_states):
            # 基底状態のビット表現 (Qiskitのエンディアンに合わせる)
            basis_bits = format(basis_idx, f"0{n_qubits}b")
            
            if self.verbose and basis_idx % 10 == 0:
                print(f"\n--- 基底状態 |{basis_bits}⟩ (idx={basis_idx}) の実振幅を測定 ---")
            
            # Hadamardテスト回路の構築
            test_circuit = QuantumCircuit(n_qubits + 1, 1)
            aux_qubit = n_qubits  # 補助qubitは最後
            
            # Step 1: 補助qubitにHadamardゲート
            test_circuit.h(aux_qubit)
            
            # Step 2: 制御された状態準備
            # |0>_a 状態 -> システムを |basis_idx> に準備
            # |1>_a 状態 -> システムに |ψ> を準備
            
            # --- 制御-U_i (Control=0) ---
            # |basis_idx> を準備する (Xゲートのセット)
            for qubit_i in range(n_qubits):
                if basis_bits[::-1][qubit_i] == "1":
                    test_circuit.mcx([aux_qubit], qubit_i, ctrl_state="0")
            
            # --- 制御-U_ψ (Control=1) ---
            # Step 2a: 制御-U_b (|b>の準備)
            prep_b_gate = StatePreparation(b_normalized)
            c_prep_b_gate = prep_b_gate.control(1)
            test_circuit.append(c_prep_b_gate, [aux_qubit] + sys_wires)
            
            # Step 2b: 制御-U_QSVT
            c_qsvt_gate = qsvt_circuit.control(1)
            test_circuit.append(c_qsvt_gate, [aux_qubit] + sys_qubits)
            
            # Step 3: 補助qubitに再度Hadamardゲート
            test_circuit.h(aux_qubit)
            
            if self.use_statevector:
                # Statevectorモード: より正確
                statevector = Statevector.from_instruction(test_circuit)
                all_probs = statevector.probabilities()
                cutoff_index = 2**n_qubits
                # 補助ビットが 0 の成分の確率
                p0 = np.sum(all_probs[:cutoff_index])
                # 補助ビットが 1 の成分の確率
                p1 = np.sum(all_probs[cutoff_index:])
                real_amplitudes[basis_bits] = p0 - p1
            else:
                # 測定モード: 実デバイス向け
                test_circuit.measure(aux_qubit, 0)
                
                transpiled_circuit = transpile(test_circuit, backend)
                result = backend.run(transpiled_circuit, shots=SHOTS).result()
                counts = result.get_counts()
                
                # 補助qubitの測定結果のみを使用
                aux_0 = counts.get("0", 0)
                aux_1 = counts.get("1", 0)
                total = aux_0 + aux_1
                if total > 0:
                    p0 = aux_0 / total
                    p1 = aux_1 / total
                    # Hadamardテストの結果: Re(⟨basis_idx|ψ⟩) = P(0) - P(1)
                    real_part = p0 - p1
                    real_amplitudes[basis_bits] = real_part
                else:
                    real_amplitudes[basis_bits] = 0.0
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("測定結果まとめ")
            print("=" * 60)
            print("全基底状態の期待値の実部（主要なもののみ）:")
            for i in range(min(10, n_total_states)):
                bits = format(i, f"0{n_qubits}b")
                print(f"  |{bits}⟩: {real_amplitudes.get(bits, 0):.4f}")
            print("=" * 60)
        
        return real_amplitudes
    
    def _reconstruct_from_real_amplitudes(self, real_amplitudes: Dict[str, float]) -> np.ndarray:
        """
        最新のqsvtブランチの実装に合わせて、Hadamardテストの結果から解を再構成
        
        Args:
            real_amplitudes: Hadamardテストの結果（全基底状態の実振幅）
            
        Returns:
            解ベクトル
        """
        # anc=0 (左端が0) のカウントだけ抜く
        post_amplitudes = {
            bits: c for bits, c in real_amplitudes.items() if bits[0] == "0"
        }
        
        if self.verbose:
            print("post_amplitudes:", {k: f"{v:.4f}" for k, v in list(post_amplitudes.items())[:10]})
        
        # システムの振幅を抽出
        amp_sys = np.zeros(self.matrix_size, dtype=float)
        for bits, c in post_amplitudes.items():
            sys_bit = bits[1:]  # 右側がsystem
            idx = int(sys_bit, 2) if sys_bit else 0
            if idx < self.matrix_size:
                amp_sys[idx] = c
        
        if self.verbose:
            print("amp_sys:", amp_sys)
        
        return amp_sys
    
    def _reconstruct_from_counts(self, counts: dict, shots: int) -> np.ndarray:
        """
        測定結果から解ベクトルを再構成（後方互換性のため残す）
        
        Args:
            counts: 測定結果のカウント
            shots: ショット数
            
        Returns:
            解ベクトル（振幅の絶対値）
        """
        # anc=0のカウントだけ抽出
        # NOTE: Qiskitの仕様上最初のビットが最右に記載される
        post_counts = {bits: c for bits, c in counts.items() if bits[-1] == "0"}
        total_post = sum(post_counts.values())
        
        if total_post == 0:
            raise RuntimeError("anc=0 のデータがありません。")
        
        # システムの確率（現在は2×2限定）
        if self.matrix_size == 2:
            p_sys = [0.0, 0.0]
            for bits, c in post_counts.items():
                sys_bit = bits[0]  # 左側がsystem
                p_sys[int(sys_bit)] += c
            
            # 元の実装に合わせてSHOTSで正規化（条件付き確率ではなく全体に対する確率）
            p_sys = [p / shots for p in p_sys]
            
            # 振幅の大きさ
            solution = np.sqrt(p_sys)
            return solution
        else:
            raise NotImplementedError(
                f"Reconstruction for {self.matrix_size}×{self.matrix_size} "
                "matrices is not yet implemented."
            )

