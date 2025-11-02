"""
Qiskit implementation of Quantum Singular Value Transformation (QSVT)
Based on PennyLane's qml.qsvt implementation
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def block_encode(matrix, wires):
    """
    Block encode a matrix into a unitary operator.

    Args:
        matrix: Matrix to encode (can be non-square)
        wires: List of qubit indices to use

    Returns:
        QuantumCircuit with the block encoding
    """
    matrix = np.asarray(matrix)
    m, n = matrix.shape if matrix.ndim > 1 else (1, matrix.size)

    # Determine the size needed for block encoding
    max_dim = max(m, n)
    num_qubits = len(wires)
    block_size = 2**num_qubits

    if max_dim > block_size:
        raise ValueError(
            f"Matrix dimension {max_dim} requires more than {num_qubits} qubits"
        )

    # Create block encoded unitary
    U = np.zeros((block_size, block_size), dtype=complex)

    # Place matrix in top-left corner
    U[:m, :n] = matrix

    # Complete to unitary using SVD approach
    # Fill remaining entries to make it unitary
    if m < block_size or n < block_size:
        # Simple completion: make orthogonal columns/rows
        for i in range(m, block_size):
            U[i, i] = 1.0

        # Gram-Schmidt orthogonalization to ensure unitarity
        from scipy.linalg import qr

        Q, R = qr(U)
        U = Q

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

    This is a simplified version - full implementation would use more
    sophisticated polynomial transformation algorithms.

    Args:
        poly_coeffs: Polynomial coefficients

    Returns:
        List of phase angles
    """
    # Simplified angle computation
    # In practice, this requires solving optimization problem
    # or using specific polynomial-to-angle conversion algorithms

    degree = len([c for c in poly_coeffs if c != 0])

    # For demonstration, use simple heuristic based on coefficients
    angles = []
    for i, coef in enumerate(poly_coeffs):
        if coef != 0:
            # Scale angle based on coefficient and position
            angle = np.arctan(coef) * (i + 1) / len(poly_coeffs)
            angles.append(angle)

    # Ensure we have right number of angles
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
    dim = 2 ** (num_qubits - 1) if num_qubits > 1 else 1

    for i, angle in enumerate(angles):
        # Apply projector-controlled phase
        if i == 0:
            # Initial phase
            qc_phase = pc_phase(angle, dim, wires)
            qc.compose(qc_phase, inplace=True)

        # Apply block encoding (or its adjoint)
        if i < len(angles) - 1:
            if i % 2 == 0:
                qc.compose(U_A, inplace=True)
            else:
                qc.compose(U_A.inverse(), inplace=True)

            # Apply next phase
            qc_phase = pc_phase(angles[i + 1], dim, wires)
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
