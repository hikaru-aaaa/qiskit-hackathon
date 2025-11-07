import numpy as np
from .inverse_matrix import compute_matrix_inverse_qsvt, generate_angles_qsvt_and_scale
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.exceptions import QiskitError
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from typing import Optional

SHOTS = 8192
# NOTE: ノイズなし → statevector_simulator。ノイズ付き → qasm_simulatorにしてnoise_modelを指定。
SIMULATOR = "statevector_simulator"


class LSESolver:
    """
    連立一次方程式 Ax = b を解くためのクラス
    QSVTと古典的な方法の両方をサポート
    """

    def __init__(self, A, b, kappa=4, noise_model: Optional[NoiseModel] = None):
        self.A = np.asarray(A, dtype=complex)
        self.b = np.asarray(b, dtype=complex)
        self.n, self.m = self.A.shape
        self.frobenius_norm = np.linalg.norm(self.A, ord="fro")
        self.A_normalized = self.A / self.frobenius_norm
        self.kappa = kappa
        self.noise_model = noise_model

    def solve_lse_classical(self):
        """
        古典的な方法で連立一次方程式 Ax = b を解く
        """
        A_inv = np.linalg.inv(self.A)
        x_solution = A_inv @ self.b

        return x_solution

    def solve_linear_system_quantum(self, statevector=True):
        """
        実際の量子デバイスでの連立一次方程式の解法
        """
        # 1) b を正規化（複素数型にしておくと無難）
        b_norm = np.linalg.norm(self.b)
        b_normalized = (self.b / b_norm).astype(complex)

        # 2) QSVT回路を取得（戻り値は適宜合わせて）
        #    qsvt_circuit: ユニタリ全体）

        kappa = self.kappa
        angles_qsvt, s = generate_angles_qsvt_and_scale(kappa)

        full_unitary, qsvt_circuit, encoding_wires, _ = compute_matrix_inverse_qsvt(
            self.A_normalized, angles_qsvt
        )

        n_sys = int(np.ceil(np.log2(len(self.b))))
        sys_wires = list[int, ...](range(n_sys))

        # 3) 合成用の土台回路は “QSVT回路と同じ本数” にする
        combined_circuit = QuantumCircuit(qsvt_circuit.num_qubits)

        # 4) |b> を “系” の量子ビットにだけ初期化
        combined_circuit.initialize(b_normalized, sys_wires)

        # # 5) QSVT を合成（qubit 並びが一致していればOK）
        combined_circuit.compose(qsvt_circuit, inplace=True)

        # # Step 5: 測定で解xを取得
        # # システムレジスタのみ測定（アンシラは無視）
        # combined_circuit.measure_all()

        # # Step 6: 実行と後処理
        # backend = Aer.get_backend(SIMULATOR)
        # result = backend.run(combined_circuit, shots=SHOTS).result()
        # counts = result.get_counts()
        # print("Measurement results:", counts)

        # # Step 7: 測定結果から解ベクトルを再構成
        # x_solution = self._reconstruct_from_counts(counts)

        real_amplitudes = self.measure_real_amplitudes(
            qsvt_circuit, b_normalized, statevector
        )

        # スケールを調整
        x_solution = (
            self._reconstruct_from_counts(real_amplitudes)
            * b_norm
            / (s * self.frobenius_norm)
        )

        return x_solution, qsvt_circuit

    def measure_real_amplitudes(self, qsvt_circuit, b_normalized, statevector=False):
        """
        Hadamardテスト（Overlap Test）を用いて全基底状態の確率振幅の実部を測定

        QSVT適用後の状態 |ψ⟩ = Σᵢ αᵢ|i⟩ の各振幅の実部 Re(αᵢ) を測定
        """
        # --- システム情報の定義 ---
        # b_normalized は |b⟩ の状態ベクトル (長さ 2**n_sys)
        # qsvt_circuit は n_qubits の回路
        n_sys = int(np.ceil(np.log2(len(self.b))))
        sys_wires = list(range(n_sys))  # |b⟩ を準備するワイヤ
        n_qubits = qsvt_circuit.num_qubits  # システム全体のqubit数
        sys_qubits = list(range(n_qubits))  # QSVT回路が作用する全ワイヤ
        
        # Create backend with noise model if provided
        if self.noise_model is not None:
            # Use density matrix simulator for noisy simulation
            backend = AerSimulator(method="density_matrix", noise_model=self.noise_model)
        else:
            # Use statevector simulator for noiseless simulation
            backend = Aer.get_backend(SIMULATOR)

        # 全基底状態の数（2^n_qubits）
        n_total_states = 2**n_qubits

        real_amplitudes = {}

        # 各基底状態に対してHadamardテストを実行
        for basis_idx in range(n_total_states):
            # 基底状態のビット表現 (Qiskitのエンディアンに合わせる)
            basis_bits = format(basis_idx, f"0{n_qubits}b")

            # ========================================
            # Hadamardテスト回路の構築
            # ========================================
            # 補助qubit + QSVT回路のqubits
            test_circuit = QuantumCircuit(n_qubits + 1, 1)
            aux_qubit = n_qubits  # 補助qubitは最後

            # Step 1: 補助qubitにHadamardゲート
            # 状態: 1/sqrt(2) * (|0>_a + |1>_a) |0...0>_s
            test_circuit.h(aux_qubit)

            # ========================================
            # Step 2: 制御された状態準備
            # |0>_a 状態 -> システムを |basis_idx> に準備
            # |1>_a 状態 -> システムに |ψ> を準備
            # ========================================

            # --- 制御-U_i (Control=0) ---
            # |basis_idx> を準備する (Xゲートのセット)
            # 補助qubitが0の時に実行 (ctrl_state='0' を使用)
            for qubit_i in range(n_qubits):
                # Qiskitのインデックス (q0, q1, ...) に合わせるためビットを逆順 (basis_bits[::-1]) で読む
                if basis_bits[::-1][qubit_i] == "1":
                    # 補助qubitが0の時、Xを適用
                    test_circuit.mcx([aux_qubit], qubit_i, ctrl_state="0")

            # --- 制御-U_ψ (Control=1) ---
            # U_ψ = U_QSVT * U_b (ここで U_b は |b> を準備するゲート)
            # 両方の操作を aux_qubit=1 で制御

            # Step 2a: 制御-U_b (|b>の準備)
            # b_normalized (長さ 2^n_sys) を sys_wires (0...n_sys-1) に準備
            # StatePreparation は b_normalized のベクトルの長さに
            # 応じたqubit数 (n_sys) を要求する
            prep_b_gate = StatePreparation(b_normalized)
            c_prep_b_gate = prep_b_gate.control(1)

            # sys_wires (e.g., [0, 1]) に適用
            test_circuit.append(c_prep_b_gate, [aux_qubit] + sys_wires)

            # Step 2b: 制御-U_QSVT
            # qsvt_circuit を n_qubits 全体 (sys_qubits) に適用
            c_qsvt_gate = qsvt_circuit.control(1)
            test_circuit.append(c_qsvt_gate, [aux_qubit] + sys_qubits)

            # この時点で、状態は 1/sqrt(2) * (|0>_a |i>_s + |1>_a |ψ>_s)

            # Step 3: 補助qubitに再度Hadamardゲート
            test_circuit.h(aux_qubit)

            if statevector:
                if self.noise_model is not None:
                    # For noisy simulation, use density matrix
                    # Save density matrix to extract probabilities
                    test_circuit.save_density_matrix()
                    transpiled_circuit = transpile(test_circuit, backend)
                    result = backend.run(transpiled_circuit, shots=8192).result()
                    
                    # Extract probabilities from density matrix
                    try:
                        density_matrix = result.data().get('density_matrix')
                        if density_matrix is not None:
                            # Calculate probabilities from density matrix diagonal
                            all_probs = np.real(np.diag(density_matrix))
                        else:
                            # Fallback: try to get statevector if available
                            sv = result.get_statevector(test_circuit)
                            all_probs = np.abs(sv) ** 2
                    except Exception as e:
                        # If density matrix extraction fails, raise informative error
                        raise QiskitError(f"Could not extract probabilities from noisy simulation result: {e}")
                else:
                    # For noiseless simulation, use statevector
                    sv = Statevector.from_instruction(test_circuit)
                    all_probs = sv.probabilities()
                
                cutoff_index = 2**n_qubits
                # 補助ビットが 0 の成分の確率
                p0 = np.sum(all_probs[:cutoff_index])
                # 補助ビットが 1 の成分の確率
                p1 = np.sum(all_probs[cutoff_index:])
                real_amplitudes[f"{basis_bits}"] = p0 - p1

            else:
                # Step 4: 補助qubitのみを測定
                test_circuit.measure(aux_qubit, 0)

                print(test_circuit.draw(output="text"))

                # ========================================
                # 実行と結果取得
                # ========================================
                transpiled_circuit = transpile(test_circuit, backend)
                result = backend.run(transpiled_circuit, shots=SHOTS).result()
                counts = result.get_counts()

                # ========================================
                # 結果の解析
                # ========================================
                # 補助qubitの測定結果のみを使用
                aux_0 = counts.get("0", 0)
                aux_1 = counts.get("1", 0)
                total = aux_0 + aux_1
                p0 = aux_0 / total
                p1 = aux_1 / total

                # Hadamardテストの結果: Re(⟨basis_idx|ψ⟩) = P(0) - P(1)
                real_part = p0 - p1

                print(f"測定結果: {counts}")
                print(f"P(補助=0) = {p0:.4f}, P(補助=1) = {p1:.4f}")
                print(f"実部推定値 (Re(α_{basis_idx})): {real_part:.4f}")

                real_amplitudes[f"{basis_bits}"] = real_part

        # ========================================
        # 結果の整理
        # ========================================

        # 後処理用に全ての振幅を返す
        return real_amplitudes
        # return real_amplitudes_statevector

    def _reconstruct_from_counts(self, real_amplitudes):
        """
        現在の配置（|anc>|b> の順＝qubit0=b、qubit1=0）に対応。
        anc=0 のときの条件付き確率から振幅の絶対値を復元。
        """
        # anc=0 (左端が0) のカウントだけ抜く
        post_amplitudes = {
            bits: c for bits, c in real_amplitudes.items() if bits[0] == "0"
        }

        amp_sys = np.zeros(len(post_amplitudes))
        for bits, c in post_amplitudes.items():
            sys_bit = bits[1:]  # 右側がsystem
            amp_sys[int(sys_bit, 2)] = c

        return amp_sys

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


if __name__ == "__main__":
    statevector = True
    # A = np.array([3, 1, 1, 3]).reshape((2, 2))
    # b = np.array([1, 2], dtype="complex")

    # NOTE: より大きな行列のテスト用に残しておく。
    A = np.array(
        [
            [0.65713691, -0.05349524, 0.08024556, -0.07242864],
            [-0.05349524, 0.65713691, -0.07242864, 0.08024556],
            [0.08024556, -0.07242864, 0.65713691, -0.05349524],
            [-0.07242864, 0.08024556, -0.05349524, 0.65713691],
        ]
    )

    b = np.array([1, 2, 3, 4], dtype="complex")

    print(f"A:\n{np.round(A, 4)}")
    print(f"b:\n{np.round(b, 4)}")

    lse_solver = LSESolver(A=A, b=b)
    x_solution, _ = lse_solver.solve_linear_system_quantum(statevector=statevector)
    print(f"x_solution: {np.round(x_solution, 4)}")
    x_solution_classical = lse_solver.solve_lse_classical()
    print(f"x_solution_classical: {np.round(x_solution_classical, 4)}")
