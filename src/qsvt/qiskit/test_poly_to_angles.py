"""
Test our pure NumPy/SciPy poly_to_angles implementation
"""

import numpy as np
from poly_to_angles_qiskit import poly_to_angles as poly_to_angles_qiskit
import pennylane as qml

# Test polynomial: P(x) = -1.5x + 2.5x³
target_poly = [0, -1.5, 0, 2.5]

print("="*60)
print("Testing poly_to_angles Implementation")
print("="*60)
print(f"Polynomial: {target_poly}")
print(f"This represents: P(x) = -1.5x + 2.5x³")
print()

# Get angles from our implementation
print("Computing angles with pure NumPy/SciPy implementation...")
angles_qiskit = poly_to_angles_qiskit(target_poly, "QSVT")
print(f"Qiskit implementation: {angles_qiskit}")
print()

# Get angles from PennyLane
print("Computing angles with PennyLane...")
angles_pennylane = qml.poly_to_angles(target_poly, "QSVT")
print(f"PennyLane implementation: {angles_pennylane}")
print()

# Compare
diff = np.abs(angles_qiskit - angles_pennylane)
max_diff = np.max(diff)

print("="*60)
print("Comparison:")
print("="*60)
print(f"Maximum difference: {max_diff}")
print(f"Mean difference: {np.mean(diff)}")
print()

if max_diff < 1e-6:
    print("✓ SUCCESS: Implementations match!")
else:
    print("✗ FAILURE: Implementations differ!")
    print("\nDifferences:")
    for i, (q, p) in enumerate(zip(angles_qiskit, angles_pennylane)):
        print(f"  Angle {i}: Qiskit={q:10.6f}, PennyLane={p:10.6f}, Diff={abs(q-p):.2e}")

# Test with another polynomial
print("\n" + "="*60)
print("Testing with another polynomial")
print("="*60)

poly2 = [0, 1.0, 0, -0.5, 0, 1/3]  # P(x) = x - x³/2 + x⁵/3
print(f"Polynomial: {poly2}")
print(f"This represents: P(x) = x - 0.5x³ + 0.333x⁵")
print()

angles_q2 = poly_to_angles_qiskit(poly2, "QSVT")
angles_p2 = qml.poly_to_angles(poly2, "QSVT")

print(f"Qiskit: {angles_q2}")
print(f"PennyLane: {angles_p2}")

diff2 = np.abs(angles_q2 - angles_p2)
max_diff2 = np.max(diff2)
print(f"\nMaximum difference: {max_diff2}")

if max_diff2 < 1e-6:
    print("✓ SUCCESS: Second test also matches!")
else:
    print("✗ FAILURE: Second test differs!")

