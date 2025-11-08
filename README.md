# Quantum Linear System Solvers: QSVT and DF-VQLS

This repository implements and compares two quantum algorithms for solving linear systems **Ax = b**:

- **QSVT** (Quantum Singular Value Transformation)
- **DF-VQLS** (Decomposition-Free Variational Quantum Linear Solver)

Built with Qiskit for quantum circuit simulation and real device deployment.

---

## What is QSVT?

**Quantum Singular Value Transformation** is a powerful and unified quantum framework that allows for the construction of quantum circuits to apply arbitrary polynomial transformations to the singular values of a matrix.

### How QSVT Works

1. **Block Encoding**: Embed matrix A into a unitary operator U
2. **Polynomial Approximation**: Apply polynomial p(x) ≈ 1/x to approximate A⁻¹
3. **Quantum Circuit**: Construct QSVT circuit using the approximation

**Key Formula:**

```
|ψ⟩ ≈ p(A)|b⟩ ≈ A⁻¹|b⟩ = |x⟩
```

### QSVT Characteristics

**Strengths:**

- Direct computation (no iterative optimization)
- Theoretical polynomial quantum speedup

**Limitations:**

- Lower precision in practice (~10⁻⁴)
- Block encoding overhead

---

## What is DF-VQLS?

**Decomposition-Free Variational Quantum Linear Solver** is a hybrid quantum-classical algorithm that finds solutions through iterative optimization, without requiring matrix decomposition.

### How DF-VQLS Works

1. **Vectorization**: Convert matrix A to vector |vec(A)⟩ via amplitude encoding
2. **Ansatz**: Prepare trial solution |u(θ)⟩ = V(θ)|0⟩ using parameterized circuit
3. **Swap Test**: Compute cost function via quantum state overlap measurement
4. **Optimization**: Classical optimizer (COBYLA) updates parameters θ to minimize cost
5. **Convergence**: Iterate until cost C(θ) ~ 0

**Cost Function:**

```
C(θ) = 1 - |⟨b|A|u(θ)⟩|² / ⟨u(θ)|A†A|u(θ)⟩
```

### DF-VQLS Characteristics

**Strengths:**

- No matrix decomposition required
- Flexible ansatz can adapt to problem structure

**Limitations:**

- Iterative optimization (many evaluations needed)
- Amplitude encoding creates quantum input problem (Could be mitigated with q-RAMs)
- Barren plateaus for large systems

## References

### Academic Papers

- **QSVT**: [Gilyén et al., "Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics"](https://dl.acm.org/doi/abs/10.1145/3313276.3316366)
- **DF-VQLS**: [Yongchun et al., "Decomposition-free variational quantum linear solver: Application in computational mechanics" ](https://www.researchgate.net/publication/395543515_Decomposition-free_variational_quantum_linear_solver_Application_in_computational_mechanics?enrichId=rgreq-7ba3ccd9b6f798688d5342cc94e0bf49-XXX&enrichSource=Y292ZXJQYWdlOzM5NTU0MzUxNTtBUzoxMTQzMTI4MTYzNzIwOTExNUAxNzU4MDg4OTYwNzQ5)
