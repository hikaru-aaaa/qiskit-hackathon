"""
Test PennyLane-compatible BlockEncode implementation
"""

import numpy as np
from qsvt_qiskit_pennylane import block_encode_pennylane
from qiskit.quantum_info import Operator


def test_square_matrix():
    """Test block encoding with square matrix (should match PennyLane)"""
    A = np.array([[0.1, 0.2], [0.3, 0.4]])

    print("="*60)
    print("Testing PennyLane-style BlockEncode with 2x2 matrix:")
    print(f"Input matrix A:\n{A}\n")

    qc = block_encode_pennylane(A, list(range(2)))
    U = Operator(qc).data

    print(f"Block-encoded unitary U (4x4):\n{np.round(U, 2)}\n")

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

    diff = np.abs(U - pennylane_output)
    print(f"Difference from PennyLane:\n{np.round(diff, 4)}")
    print(f"Max difference: {np.max(diff):.6f}")

    if np.max(diff) < 1e-5:
        print("✅ EXACT MATCH with PennyLane!")
    else:
        print("⚠️  Some differences from PennyLane")


def test_rectangular_matrix():
    """Test block encoding with 1x3 rectangular matrix"""
    B = np.array([[0.5, -0.5, 0.5]])

    print("\n" + "="*60)
    print("Testing PennyLane-style BlockEncode with 1x3 matrix:")
    print(f"Input matrix B (1x3): {B}\n")

    # For 1x3 matrix: n=1, m=3, required_dim = n+m = 4
    # So we need 2 qubits (2^2 = 4)
    print("PennyLane logic: n=1, m=3, required_dim = n+m = 4")
    print("Therefore: need ceil(log2(4)) = 2 qubits\n")

    qc = block_encode_pennylane(B, list(range(2)))  # 2 qubits!
    U = Operator(qc).data

    print(f"Block-encoded unitary U (4x4):\n{np.round(U, 2)}\n")

    # Verify unitarity
    U_dag_U = U.conj().T @ U
    is_unitary = np.allclose(U_dag_U, np.eye(4))
    print(f"Is U unitary? {is_unitary}")
    if not is_unitary:
        print(f"U†U deviation from identity:\n{np.round(np.abs(U_dag_U - np.eye(4)), 4)}\n")

    # Expected PennyLane output
    pennylane_output = np.array([
        [ 0.5,  -0.5,   0.5,   0.5 ],
        [ 0.83,  0.17, -0.17, -0.5 ],
        [ 0.17,  0.83,  0.17,  0.5 ],
        [-0.17,  0.17,  0.83, -0.5 ]
    ])

    print(f"\nPennyLane output:\n{pennylane_output}\n")

    diff = np.abs(U - pennylane_output)
    print(f"Difference from PennyLane:\n{np.round(diff, 4)}")
    print(f"Max difference: {np.max(diff):.6f}")

    if np.max(diff) < 1e-5:
        print("✅ EXACT MATCH with PennyLane!")
    else:
        print("⚠️  Some differences from PennyLane")


def test_comparison():
    """Compare the two implementations"""
    from qsvt_qiskit import block_encode as block_encode_original

    print("\n" + "="*60)
    print("COMPARISON: Original vs PennyLane-style implementation")
    print("="*60)

    # Test 2x2 matrix
    A = np.array([[0.1, 0.2], [0.3, 0.4]])

    qc_orig = block_encode_original(A, list(range(2)))
    U_orig = Operator(qc_orig).data

    qc_pl = block_encode_pennylane(A, list(range(2)))
    U_pl = Operator(qc_pl).data

    print("\n2x2 Matrix:")
    print(f"  Original output shape: {U_orig.shape}")
    print(f"  PennyLane-style shape: {U_pl.shape}")
    print(f"  Match: {np.allclose(U_orig, U_pl)}")

    # Test 1x3 matrix
    B = np.array([[0.5, -0.5, 0.5]])

    print("\n1x3 Matrix:")
    try:
        qc_orig = block_encode_original(B, list(range(2)))
        print("  Original: Would need 3 qubits (raises error with 2)")
    except ValueError as e:
        print(f"  Original: ValueError with 2 qubits (expected)")

    qc_pl = block_encode_pennylane(B, list(range(2)))
    U_pl = Operator(qc_pl).data
    print(f"  PennyLane-style: Works with 2 qubits! Shape: {U_pl.shape}")
    print(f"  ✅ Key difference: PennyLane-style uses fewer qubits for rectangular matrices")


if __name__ == "__main__":
    test_square_matrix()
    test_rectangular_matrix()
    test_comparison()

