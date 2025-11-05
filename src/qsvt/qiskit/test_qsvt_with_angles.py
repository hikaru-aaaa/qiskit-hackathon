"""
Test QSVT with corrected angle computation
"""

import numpy as np
import matplotlib.pyplot as plt
from qsvt_qiskit_pennylane import qsvt
from qiskit.quantum_info import Operator

# Target polynomial: P(x) = -3/2 * x + 5/2 * x^3
target_poly = [0, -3 * 0.5, 0, 5 * 0.5]

print("Target polynomial coefficients:", target_poly)
print("This represents: P(x) = -1.5x + 2.5x^3\n")


def qsvt_output(a):
    """Compute QSVT output for a single value"""
    # Get the QSVT circuit
    qc = qsvt(a, target_poly, encoding_wires=[0], block_encoding="embedding")
    # Get unitary matrix
    out = Operator(qc).data
    return out[0, 0]  # top-left entry


# Test on a range of values
a_vals = np.linspace(-1, 1, 50)
qsvt_vals = [np.real(qsvt_output(a)) for a in a_vals]  # neglect small imaginary part
target = [np.polyval(target_poly[::-1], a) for a in a_vals]  # evaluate polynomial

print("Testing QSVT transformation:")
print(f"a = 0.0: QSVT = {np.real(qsvt_output(0.0)):.4f}, Target = {np.polyval(target_poly[::-1], 0.0):.4f}")
print(f"a = 0.5: QSVT = {np.real(qsvt_output(0.5)):.4f}, Target = {np.polyval(target_poly[::-1], 0.5):.4f}")
print(f"a = 1.0: QSVT = {np.real(qsvt_output(1.0)):.4f}, Target = {np.polyval(target_poly[::-1], 1.0):.4f}")
print(f"a = -1.0: QSVT = {np.real(qsvt_output(-1.0)):.4f}, Target = {np.polyval(target_poly[::-1], -1.0):.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(a_vals, target, 'b-', label="Target P(x) = -1.5x + 2.5x³", linewidth=2)
plt.plot(a_vals, qsvt_vals, 'r*', label="QSVT", markersize=6)

# Mark specific points
plt.axhline(y=0.75, color='g', linestyle='--', alpha=0.5)
plt.axhline(y=-0.75, color='g', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='g', linestyle='--', alpha=0.5)
plt.axvline(x=-1, color='g', linestyle='--', alpha=0.5)

plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('QSVT: Polynomial Transformation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('qsvt_test.png', dpi=150)
print("\nPlot saved as 'qsvt_test.png'")

plt.show()

# Calculate error
error = np.mean(np.abs(np.array(qsvt_vals) - np.array(target)))
max_error = np.max(np.abs(np.array(qsvt_vals) - np.array(target)))
print(f"\nMean absolute error: {error:.6f}")
print(f"Max absolute error: {max_error:.6f}")

if max_error < 0.01:
    print("✓ QSVT successfully matches target polynomial!")
else:
    print("✗ QSVT does not match target polynomial accurately")

