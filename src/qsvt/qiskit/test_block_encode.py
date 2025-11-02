"""
Test script to verify block encoding implementation
"""

import numpy as np
from qsvt_qiskit import block_encode
from qiskit.quantum_info import Operator


def test_square_matrix():
    """Test block encoding with the example from PennyLane"""
    A = np.array([[0.1, 0.2], [0.3, 0.4]])

    print("Testing BlockEncode with square matrix:")
    print(f"Input matrix A:\n{A}\n")

    qc = block_encode(A, list(range(2)))
    U = Operator(qc).data

    print(f"Block-encoded unitary U:\n{np.round(U, 2)}\n")
    print(f"Top-left 2x2 block (should be A):\n{np.round(U[:2, :2], 2)}\n")

    # Verify unitarity
    U_dag_U = U.conj().T @ U
    is_unitary = np.allclose(U_dag_U, np.eye(4))
    print(f"Is U unitary? {is_unitary}")
    if not is_unitary:
        print(f"U†U deviation from identity:\n{np.round(np.abs(U_dag_U - np.eye(4)), 4)}\n")

    # Expected PennyLane output
    pennylane_output = np.array([
        [ 0.1,   0.2,   0.97, -0.06],
        [ 0.3,   0.4,  -0.06,  0.86],
        [ 0.95, -0.08, -0.1,  -0.3 ],
        [-0.08,  0.89, -0.2,  -0.4 ]
    ])

    print(f"\nPennyLane output:\n{pennylane_output}\n")
    print(f"Difference from PennyLane:\n{np.round(np.abs(U - pennylane_output), 2)}")


def test_rectangular_matrix():
    """Test block encoding with rectangular matrix"""
    B = np.array([[0.5, -0.5, 0.5]])

    print("\n" + "="*60)
    print("Testing BlockEncode with rectangular matrix:")
    print(f"Input matrix B (1x3):\n{B}\n")

    # For 1x3 matrix, padded to 3x3, block encoding becomes 6x6
    # Need at least 3 qubits (2^3 = 8 >= 6)
    qc = block_encode(B, list(range(3)))
    U = Operator(qc).data

    print(f"Block-encoded unitary U (8x8):\n{np.round(U, 2)}\n")
    print(f"Top-left 6x6 block (actual block encoding):\n{np.round(U[:6, :6], 2)}\n")

    # Verify unitarity
    U_dag_U = U.conj().T @ U
    is_unitary = np.allclose(U_dag_U, np.eye(8))
    print(f"Is U unitary? {is_unitary}")
    if not is_unitary:
        print(f"U†U deviation from identity:\n{np.round(np.abs(U_dag_U - np.eye(8)), 4)}\n")

    # Note: PennyLane uses 2 qubits (4x4) which suggests a different encoding scheme
    # Our implementation uses standard block encoding which requires more space
    print("\nNote: PennyLane's output uses 2 qubits (4x4), suggesting a more compact")
    print("encoding scheme. Our standard block encoding uses 3 qubits (8x8) for a")
    print("1x3 matrix to properly implement the mathematical block encoding formula.")


if __name__ == "__main__":
    test_square_matrix()
    test_rectangular_matrix()

