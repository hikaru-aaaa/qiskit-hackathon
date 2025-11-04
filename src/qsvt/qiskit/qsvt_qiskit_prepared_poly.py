"""
pennylaneで用意されているアングルを用いて、QSVTを実行する。
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm


def block_encode_pennylane(matrix, wires):
    """
    Block encode a matrix following PennyLane's exact algorithm.

    This is a direct port of PennyLane's BlockEncode from
    ops/qubit/matrix_ops.py (lines 695-937).

    For a matrix A (n x m), creates a unitary U such that:
    U = [[A,              sqrt(I - A†A)     ],
         [sqrt(I - AA†),  -A†              ]]

    If the operator norm of A is greater than 1, it is normalized.
    The required Hilbert space dimension is (n + m).

    Args:
        matrix: Matrix to encode (can be non-square, 1D, or scalar)
        wires: List of qubit indices to use

    Returns:
        QuantumCircuit with the block encoding

    Reference:
        PennyLane BlockEncode: ops/qubit/matrix_ops.py, lines 695-937
    """
    # Convert to numpy array
    A = np.asarray(matrix, dtype=complex)

    # Handle different input shapes (following PennyLane logic)
    shape_a = A.shape
    num_qubits = len(wires)
    hilbert_dim = 2**num_qubits

    # Handle scalar or very small matrices
    if shape_a == () or all(x == 1 for x in shape_a):
        A = np.reshape(A, (1, 1))
        normalization = np.abs(A[0, 0])
        n, m = 1, 1
    else:
        # Handle 1D arrays (convert to row vector, like PennyLane)
        if len(shape_a) == 1:
            A = np.reshape(A, (1, len(A)))
            shape_a = A.shape

        n, m = shape_a  # n = rows, m = cols

        # Compute normalization factor (following PennyLane)
        # max of ||AA†||_∞ and ||A†A||_∞
        AA_dag = A @ A.conj().T
        A_dag_A = A.conj().T @ A
        norm_AA = np.linalg.norm(AA_dag, ord=np.inf)
        norm_ATA = np.linalg.norm(A_dag_A, ord=np.inf)
        normalization = max(norm_AA, norm_ATA)

    # Normalize if norm > 1 (to ensure unitarity)
    # PennyLane: A / max(normalization, 1.0)
    A = A / max(normalization, 1.0)

    # Check if Hilbert space is large enough
    # PennyLane requires: hilbert_dim >= (n + m)
    required_dim = n + m
    if hilbert_dim < required_dim:
        raise ValueError(
            f"Block encoding a ({n} x {m}) matrix requires a Hilbert space of size "
            f"at least ({required_dim} x {required_dim}). "
            f"Cannot be embedded in a {num_qubits} qubit system (size {hilbert_dim})."
        )

    # Build the block-encoded unitary following PennyLane's _process_blockencode
    if n == 1 and m == 1:
        # Special case for 1x1 matrices (scalars)
        # Following PennyLane lines 907-910
        a = A[0, 0]
        sqrt_term = np.sqrt(1 - a * np.conj(a))

        # Build 2x2 block encoding
        col1 = np.array([[a], [sqrt_term]], dtype=complex)
        col2 = np.array([[sqrt_term], [-np.conj(a)]], dtype=complex)
        U = np.hstack([col1, col2])
    else:
        # General case for n x m matrices
        # Following PennyLane lines 911-929
        # where d1=n (rows), d2=m (cols)

        # Build column 1: [A; sqrt(I_m - A†A)]
        # A is n×m, sqrt(I_m - A†A) is m×m
        # Result: (n+m)×m
        I_m = np.eye(m, dtype=complex)
        sqrt_I_m_ATA = sqrtm(I_m - A.conj().T @ A)
        col1 = np.vstack([A, sqrt_I_m_ATA])

        # Build column 2: [sqrt(I_n - AA†); -A†]
        # sqrt(I_n - AA†) is n×n, -A† is m×n
        # Result: (n+m)×n
        I_n = np.eye(n, dtype=complex)
        sqrt_I_n_AAT = sqrtm(I_n - A @ A.conj().T)
        col2 = np.vstack([sqrt_I_n_AAT, -A.conj().T])

        # Combine: U = [col1 | col2]
        # Result: (n+m)×(m+n) = (n+m)×(n+m) square matrix
        U = np.hstack([col1, col2])

    # Pad with identity if needed (following PennyLane lines 931-935)
    current_size = U.shape[0]
    if current_size < hilbert_dim:
        # Create full-size identity matrix
        U_full = np.eye(hilbert_dim, dtype=complex)
        # Place U in top-left corner
        U_full[:current_size, :current_size] = U
        U = U_full

    print("block_encode_pennylane U:\n", np.round(U, 4))
    # Create quantum circuit
    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, wires, label="BlockEncode")

    return qc


def pc_phase(phi, dim, wires):
    """
    Projector-Controlled Phase gate.

    Applies phase phi to subspace spanned by first 'dim' computational basis states.

    Args:
        phi: Phase angle
        dim: Dimension of subspace
        wires: List of qubit indices

    Returns:
        QuantumCircuit with PCPhase operation
    """
    num_qubits = len(wires)
    size = 2**num_qubits

    # Create diagonal unitary with phases
    U = np.eye(size, dtype=complex)

    # Apply phase to first 'dim' basis states
    for i in range(dim):
        U[i, i] = np.exp(1j * phi)

    # Apply conjugate phase to remaining states
    for i in range(dim, size):
        U[i, i] = np.exp(-1j * phi)

    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, wires, label="PCPhase")

    return qc


def qsvt(matrix_or_value, angles, encoding_wires, block_encoding="embedding"):
    """
    Quantum Singular Value Transformation using PennyLane's BlockEncode.

    Args:
        matrix_or_value: Either a scalar value or matrix to transform
        poly_coeffs: Polynomial coefficients for transformation
        encoding_wires: Wires/qubits for encoding
        block_encoding: Type of block encoding ("embedding" or other)

    Returns:
        QuantumCircuit implementing QSVT
    """
    # Handle scalar input
    if np.isscalar(matrix_or_value):
        # Create 1x1 matrix
        matrix = np.array([[matrix_or_value]])
        num_qubits = max(1, len(encoding_wires))
    else:
        matrix = np.asarray(matrix_or_value)
        # Determine number of qubits needed using PennyLane's logic
        shape_a = matrix.shape
        if len(shape_a) == 1:
            n, m = 1, len(matrix)
        else:
            n, m = shape_a
        required_dim = n + m
        num_qubits = int(np.ceil(np.log2(required_dim)))

    # Create circuit
    if len(encoding_wires) < num_qubits:
        wires = list(range(num_qubits))
    else:
        wires = encoding_wires[:num_qubits]

    qc = QuantumCircuit(num_qubits)

    # Get block encoding of matrix using PennyLane's algorithm
    U_A = block_encode_pennylane(matrix, wires)

    # QSVT sequence: alternate between projector phases and block encoding
    # Following PennyLane's compute_decomposition logic:
    # for idx, op in enumerate(projectors[:-1]):
    #     op_list.append(op)
    #     if idx % 2 == 0:
    #         op_list.append(UA)
    #     else:
    #         op_list.append(ops.adjoint(UA_adj))
    # op_list.append(projectors[-1])

    # Determine dimension for PCPhase
    # For scalar values, dim should be 1
    # For matrices, dim alternates between row and column dimension
    if np.isscalar(matrix_or_value):
        # Scalar case: all PCPhase operations use dim=1
        dims = [1] * len(angles)
    else:
        # Matrix case: dim alternates between columns (m) and rows (n)
        shape_a = matrix.shape if matrix.ndim > 1 else (1, matrix.size)
        n, m = shape_a
        # Based on PennyLane's _tensorlike_process:
        # dim = c if idx % 2 else r (where c=rows, r=columns)
        dims = [m if i % 2 == 0 else n for i in range(len(angles))]

    # Build QSVT circuit following PennyLane's pattern
    phase_matrices = []
    for idx in range(len(angles) - 1):
        # Add projector
        qc_phase = pc_phase(angles[idx], dims[idx], wires)
        # 行列として出力
        phase_matrix = Operator(qc_phase).data
        phase_matrices.append([angles[idx], phase_matrix])
        qc.compose(qc_phase, inplace=True)

        # Add block encoding or its adjoint
        if idx % 2 == 0:
            qc.compose(U_A, inplace=True)  # U
        else:
            qc.compose(U_A.inverse(), inplace=True)  # U†

    # Add final projector
    qc_phase = pc_phase(angles[-1], dims[-1], wires)
    qc.compose(qc_phase, inplace=True)

    return qc, phase_matrices
