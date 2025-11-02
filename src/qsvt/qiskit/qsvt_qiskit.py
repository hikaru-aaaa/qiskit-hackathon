"""
Qiskit implementation of Quantum Singular Value Transformation (QSVT)
Based on PennyLane's qml.qsvt implementation
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def block_encode(matrix, wires):
    """
    Block encode a matrix into a unitary operator using the standard block encoding scheme.

    For a matrix A (m x n), this creates a unitary U such that:
    U = [[A,      B    ],
         [C,      D    ]]
    where B, C, D are chosen to make U unitary.

    Args:
        matrix: Matrix to encode (can be non-square)
        wires: List of qubit indices to use

    Returns:
        QuantumCircuit with the block encoding
    """
    matrix = np.asarray(matrix, dtype=complex)

    # Handle scalar or 1D array
    if matrix.ndim == 0:
        matrix = np.array([[matrix]])
    elif matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    m, n = matrix.shape

    # Determine the size needed for block encoding
    num_qubits = len(wires)
    block_size = 2**num_qubits

    # Block encoding creates a 2x2 block structure, so we need 2*size <= block_size
    max_dim = max(m, n)

    # The block encoded matrix will be 2*max_dim x 2*max_dim
    if 2 * max_dim > block_size:
        raise ValueError(
            f"Matrix dimension {max_dim} requires block encoding size {2 * max_dim}, "
            f"but only {block_size} available with {num_qubits} qubits. "
            f"Need at least {int(np.ceil(np.log2(2 * max_dim)))} qubits."
        )

    # Pad matrix to square if needed
    if m != n:
        size = max(m, n)
        A_padded = np.zeros((size, size), dtype=complex)
        A_padded[:m, :n] = matrix
        matrix = A_padded
        m = n = size

    # Standard block encoding algorithm
    # U = [[A,              sqrt(I - A*A^†)  ],
    #      [sqrt(I - A^†*A), -A^†            ]]

    # Compute A*A^† and A^†*A
    AA_dag = matrix @ matrix.conj().T
    A_dag_A = matrix.conj().T @ matrix

    # Compute sqrt(I - A*A^†) and sqrt(I - A^†*A)
    # Using eigenvalue decomposition for matrix square root
    from scipy.linalg import sqrtm

    I_m = np.eye(m, dtype=complex)

    # Compute complementary matrices
    try:
        B = sqrtm(I_m - AA_dag)  # sqrt(I - A*A^†)
        C = sqrtm(I_m - A_dag_A)  # sqrt(I - A^†*A)
    except (ValueError, np.linalg.LinAlgError):
        # If sqrt fails, use pseudo-inverse approach
        eigvals_AA = np.linalg.eigvalsh(AA_dag)
        eigvals_ATA = np.linalg.eigvalsh(A_dag_A)

        # Ensure eigenvalues are <= 1 (numerical stability)
        if np.any(eigvals_AA > 1.0001) or np.any(eigvals_ATA > 1.0001):
            # Normalize A if needed
            max_eigval = max(np.max(eigvals_AA), np.max(eigvals_ATA))
            matrix = matrix / np.sqrt(max_eigval + 1e-10)
            AA_dag = matrix @ matrix.conj().T
            A_dag_A = matrix.conj().T @ matrix

        B = sqrtm(I_m - AA_dag)
        C = sqrtm(I_m - A_dag_A)

    D = -matrix.conj().T  # -A^†

    # Construct the block-encoded unitary (2m x 2m for square matrix)
    U_block = np.block([[matrix, B], [C, D]])

    # Pad to full block_size if needed
    if 2 * m < block_size:
        U = np.eye(block_size, dtype=complex)
        U[: 2 * m, : 2 * m] = U_block
    else:
        U = U_block

    # Ensure the result is as close to unitary as possible
    # (numerical cleanup)
    from scipy.linalg import polar

    U_unitary, _ = polar(U)
    U = U_unitary

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


def compute_qsvt_angles(poly_coeffs):
    """
    Compute the phase angles for QSVT from polynomial coefficients.

    Uses PennyLane's poly_to_angles function for accurate angle computation.

    Args:
        poly_coeffs: Polynomial coefficients

    Returns:
        List of phase angles
    """
    try:
        # Use PennyLane's poly_to_angles for accurate computation
        import pennylane as qml

        angles = qml.poly_to_angles(poly_coeffs, "QSVT", angle_solver="root-finding")
        return np.asarray(angles)
    except ImportError:
        # Fallback: simplified angle computation (not accurate)
        import warnings

        warnings.warn(
            "PennyLane not available. Using simplified angle computation. "
            "Results may not match expected polynomial transformation. "
            "Install PennyLane for accurate QSVT: pip install pennylane"
        )

        # Very simplified fallback - will NOT produce correct results
        degree = len([c for c in poly_coeffs if c != 0])
        angles = []
        for i, coef in enumerate(poly_coeffs):
            if coef != 0:
                angle = np.arctan(coef) * (i + 1) / len(poly_coeffs)
                angles.append(angle)

        while len(angles) < degree:
            angles.append(0.0)

        return angles


def qsvt(matrix_or_value, poly_coeffs, encoding_wires, block_encoding="embedding"):
    """
    Quantum Singular Value Transformation.

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
        # Determine number of qubits needed
        max_dim = max(matrix.shape) if matrix.ndim > 1 else matrix.size
        num_qubits = int(np.ceil(np.log2(max_dim))) + 1  # +1 for ancilla

    # Compute angles from polynomial
    angles = compute_qsvt_angles(poly_coeffs)

    # Create circuit
    if len(encoding_wires) < num_qubits:
        wires = list(range(num_qubits))
    else:
        wires = encoding_wires[:num_qubits]

    qc = QuantumCircuit(num_qubits)

    # Get block encoding of matrix
    U_A = block_encode(matrix, wires[:-1] if num_qubits > 1 else wires)

    # QSVT sequence: alternate between projector phases and block encoding
    # Following PennyLane's compute_decomposition logic

    # Determine dimension for PCPhase
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
    for idx in range(len(angles) - 1):
        # Add projector
        qc_phase = pc_phase(angles[idx], dims[idx], wires)
        qc.compose(qc_phase, inplace=True)

        # Add block encoding or its adjoint
        if idx % 2 == 0:
            qc.compose(U_A, inplace=True)  # U
        else:
            qc.compose(U_A.inverse(), inplace=True)  # U†

    # Add final projector
    qc_phase = pc_phase(angles[-1], dims[-1], wires)
    qc.compose(qc_phase, inplace=True)

    return qc


def qsvt_matrix_transform(matrix, poly_coeffs, encoding_wires=None):
    """
    Apply QSVT to transform a matrix's singular values according to polynomial.

    Args:
        matrix: Input matrix
        poly_coeffs: Polynomial coefficients
        encoding_wires: Wires for encoding (auto-determined if None)

    Returns:
        Transformed matrix (top-left block of output unitary)
    """
    matrix = np.asarray(matrix)

    # Determine encoding wires
    if encoding_wires is None:
        max_dim = max(matrix.shape) if matrix.ndim > 1 else matrix.size
        num_qubits = int(np.ceil(np.log2(max_dim))) + 1
        encoding_wires = list(range(num_qubits))

    # Get QSVT circuit
    qc = qsvt(matrix, poly_coeffs, encoding_wires)

    # Get output unitary
    U_out = Operator(qc).data

    # Extract transformed matrix from top-left block
    m, n = matrix.shape if matrix.ndim > 1 else (1, matrix.size)
    transformed = U_out[:m, :n]

    return transformed
