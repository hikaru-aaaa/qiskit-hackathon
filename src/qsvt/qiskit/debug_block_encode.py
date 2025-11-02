"""
Debug block encoding
"""

import numpy as np
from scipy.linalg import sqrtm

A = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=complex)
print(f"Matrix A:\n{A}\n")

n, m = A.shape
print(f"Shape: n={n}, m={m}\n")

# Compute normalization
AA_dag = A @ A.conj().T
A_dag_A = A.conj().T @ A

print(f"AA†:\n{AA_dag}\n")
print(f"A†A:\n{A_dag_A}\n")

norm_AA = np.linalg.norm(AA_dag, ord=np.inf)
norm_ATA = np.linalg.norm(A_dag_A, ord=np.inf)
normalization = max(norm_AA, norm_ATA)

print(f"||AA†||_∞ = {norm_AA}")
print(f"||A†A||_∞ = {norm_ATA}")
print(f"normalization = {normalization}\n")

# Normalize
A = A / max(normalization, 1.0)
print(f"Normalized A:\n{A}\n")

# Recompute after normalization
AA_dag = A @ A.conj().T
A_dag_A = A.conj().T @ A

print(f"After normalization:")
print(f"AA†:\n{np.round(AA_dag, 4)}\n")
print(f"A†A:\n{np.round(A_dag_A, 4)}\n")

# Compute square roots
I_m = np.eye(m, dtype=complex)
I_n = np.eye(n, dtype=complex)

print(f"I - A†A:\n{np.round(I_m - A_dag_A, 4)}\n")
print(f"I - AA†:\n{np.round(I_n - AA_dag, 4)}\n")

sqrt_I_ATA = sqrtm(I_m - A_dag_A)
sqrt_I_AAT = sqrtm(I_n - AA_dag)

print(f"sqrt(I - A†A):\n{np.round(sqrt_I_ATA, 4)}\n")
print(f"sqrt(I - AA†):\n{np.round(sqrt_I_AAT, 4)}\n")

# Build block matrix
col1 = np.vstack([A, sqrt_I_AAT])
col2 = np.vstack([sqrt_I_ATA, -A.conj().T])

print(f"col1 shape: {col1.shape}")
print(f"col2 shape: {col2.shape}")

U = np.hstack([col1, col2])

print(f"\nU:\n{np.round(U, 4)}\n")

# Check unitarity
U_dag_U = U.conj().T @ U
print(f"U†U:\n{np.round(U_dag_U, 4)}\n")

is_unitary = np.allclose(U_dag_U, np.eye(4))
print(f"Is unitary: {is_unitary}")
print(f"Max deviation: {np.max(np.abs(U_dag_U - np.eye(4)))}")

