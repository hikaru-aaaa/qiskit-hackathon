"""
Variational Quantum Linear Solver (VQLS) Implementation

This module implements VQLS algorithm using the Hadamard test technique to solve
linear systems of equations Ax = b on quantum computers. The algorithm uses a
variational approach to find the quantum state |x⟩ that approximates the solution.

The implementation uses:
- Hadamard test for computing expectation values ⟨ψ|U|ψ⟩
- Variational ansatz with parameterized RY gates and CZ entanglement
- Classical optimizer (COBYLA) to minimize the cost function
- Qiskit Aer simulator for quantum circuit simulation

References:
    C. Bravo-Prieto et al., "Variational Quantum Linear Solver"
    arXiv:1909.05820 (2019)
"""

from typing import List, Tuple, Union
import numpy as np
import random
from scipy.optimize import minimize, OptimizeResult
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Constants for qubit indices
ANCILLA_QUBIT = 0
WORK_QUBITS = [1, 2, 3]
AUXILIARY_QUBIT = 4
NUM_QUBITS = 5


def apply_fixed_ansatz(
    circ: QuantumCircuit, qubits: List[int], parameters: List[List[float]]
) -> QuantumCircuit:
    """
    Apply a fixed hardware-efficient ansatz to the quantum circuit.

    The ansatz consists of three layers of parameterized RY rotations
    interspersed with CZ entanglement gates in a specific pattern.

    Args:
        circ: Quantum circuit to apply ansatz to
        qubits: List of qubit indices to apply the ansatz on
        parameters: 3-layer parameters, each layer has len(qubits) parameters
                   Shape: [layer][qubit_idx]

    Returns:
        Modified quantum circuit with ansatz applied
    """
    # First layer: RY rotations + CZ entanglement
    for qubit_idx in range(len(qubits)):
        circ.ry(parameters[0][qubit_idx], qubits[qubit_idx])

    circ.cz(qubits[0], qubits[1])
    circ.cz(qubits[2], qubits[0])

    # Second layer: RY rotations + CZ entanglement
    for qubit_idx in range(len(qubits)):
        circ.ry(parameters[1][qubit_idx], qubits[qubit_idx])

    circ.cz(qubits[1], qubits[2])
    circ.cz(qubits[2], qubits[0])

    # Third layer: RY rotations (final layer)
    for qubit_idx in range(len(qubits)):
        circ.ry(parameters[2][qubit_idx], qubits[qubit_idx])

    return circ


def control_fixed_ansatz(
    circ: QuantumCircuit,
    qubits: List[int],
    parameters: List[List[float]],
    control_qubit: int,
    reg: Union[int, None] = None,
) -> QuantumCircuit:
    """
    Apply controlled version of the fixed ansatz using an ancilla qubit.

    This creates a controlled-U operation where the ansatz is applied only
    when the control qubit is |1⟩. Uses Toffoli gates with auxiliary qubit
    to implement controlled-CZ operations.

    Args:
        circ: Quantum circuit to apply controlled ansatz to
        qubits: List of work qubit indices
        parameters: 3-layer parameters for the ansatz
        control_qubit: Index of the control qubit (ancilla)
        reg: Quantum register (unused, kept for backward compatibility)

    Returns:
        Modified quantum circuit with controlled ansatz applied
    """
    # First layer: Controlled RY rotations + Controlled CZ gates
    for qubit_idx in range(len(qubits)):
        circ.cry(parameters[0][qubit_idx], control_qubit, qubits[qubit_idx])

    # Implement controlled-CZ(qubits[0], qubits[1]) using auxiliary qubit
    circ.ccx(control_qubit, qubits[1], AUXILIARY_QUBIT)
    circ.cz(qubits[0], AUXILIARY_QUBIT)
    circ.ccx(control_qubit, qubits[1], AUXILIARY_QUBIT)

    # Implement controlled-CZ(qubits[2], qubits[0])
    circ.ccx(control_qubit, qubits[0], AUXILIARY_QUBIT)
    circ.cz(qubits[2], AUXILIARY_QUBIT)
    circ.ccx(control_qubit, qubits[0], AUXILIARY_QUBIT)

    # Second layer: Controlled RY rotations + Controlled CZ gates
    for qubit_idx in range(len(qubits)):
        circ.cry(parameters[1][qubit_idx], control_qubit, qubits[qubit_idx])

    # Implement controlled-CZ(qubits[1], qubits[2])
    circ.ccx(control_qubit, qubits[2], AUXILIARY_QUBIT)
    circ.cz(qubits[1], AUXILIARY_QUBIT)
    circ.ccx(control_qubit, qubits[2], AUXILIARY_QUBIT)

    # Implement controlled-CZ(qubits[2], qubits[0])
    circ.ccx(control_qubit, qubits[0], AUXILIARY_QUBIT)
    circ.cz(qubits[2], AUXILIARY_QUBIT)
    circ.ccx(control_qubit, qubits[0], AUXILIARY_QUBIT)

    # Third layer: Controlled RY rotations
    for qubit_idx in range(len(qubits)):
        circ.cry(parameters[2][qubit_idx], control_qubit, qubits[qubit_idx])

    return circ


def control_b(
    circ: QuantumCircuit, control_qubit: int, qubits: List[int]
) -> QuantumCircuit:
    """
    Apply controlled-Hadamard gates to prepare the |b⟩ state.

    This function applies controlled-H gates from the ancilla to all work qubits,
    effectively preparing a superposition state representing the right-hand side
    vector b of the linear system.

    Args:
        circ: Quantum circuit to modify
        control_qubit: Index of the control qubit (ancilla)
        qubits: List of target qubit indices

    Returns:
        Modified quantum circuit with controlled-H gates applied
    """
    for qubit in qubits:
        circ.ch(control_qubit, qubit)
    return circ


def hadamard_test(
    circ: QuantumCircuit,
    gate_type: List[List[int]],
    qubits: List[int],
    ancilla_index: int,
    parameters: List[List[float]],
) -> QuantumCircuit:
    """
    Implement the Hadamard test circuit for computing ⟨ψ|A†A|ψ⟩.

    The Hadamard test allows measuring the real part of ⟨ψ|U|ψ⟩ by using
    an ancilla qubit in superposition to control the application of U.

    Args:
        circ: Quantum circuit to build Hadamard test on
        gate_type: Two lists specifying which qubits get Z gates [list1, list2]
        qubits: List of work qubit indices
        ancilla_index: Index of ancilla qubit for Hadamard test
        parameters: Variational parameters for the ansatz

    Returns:
        Quantum circuit with Hadamard test implemented
    """
    # Create superposition on ancilla
    circ.h(ancilla_index)

    # Apply the ansatz to prepare |ψ⟩
    circ = apply_fixed_ansatz(circ, qubits, parameters)

    # Apply controlled Z gates based on gate_type specification (first set)
    for gate_idx in range(len(gate_type[0])):
        if gate_type[0][gate_idx] == 1:
            circ.cz(ancilla_index, qubits[gate_idx])

    # Apply controlled Z gates based on gate_type specification (second set)
    for gate_idx in range(len(gate_type[1])):
        if gate_type[1][gate_idx] == 1:
            circ.cz(ancilla_index, qubits[gate_idx])

    # Final Hadamard on ancilla to complete the test
    circ.h(ancilla_index)

    return circ


def special_hadamard_test(
    circ: QuantumCircuit,
    gate_type: List[int],
    qubits: List[int],
    ancilla_index: int,
    parameters: List[List[float]],
    reg: Union[int, None] = None,
) -> QuantumCircuit:
    """
    Implement special Hadamard test for computing ⟨b|A†|ψ⟩ terms.

    This variant of the Hadamard test includes the controlled ansatz and
    the preparation of |b⟩ state to compute overlaps needed for VQLS.

    Args:
        circ: Quantum circuit to build test on
        gate_type: List specifying which qubits get Z gates
        qubits: List of work qubit indices
        ancilla_index: Index of ancilla qubit
        parameters: Variational parameters for the ansatz
        reg: Quantum register (unused, kept for backward compatibility)

    Returns:
        Quantum circuit with special Hadamard test implemented
    """
    # Create superposition on ancilla
    circ.h(ancilla_index)

    # Apply controlled ansatz
    circ = control_fixed_ansatz(circ, qubits, parameters, ancilla_index, reg)

    # Apply controlled Z gates based on gate_type
    for gate_idx in range(len(gate_type)):
        if gate_type[gate_idx] == 1:
            circ.cz(ancilla_index, qubits[gate_idx])

    # Prepare |b⟩ state with controlled Hadamards
    circ = control_b(circ, ancilla_index, qubits)

    # Final Hadamard on ancilla
    circ.h(ancilla_index)

    return circ


def extract_odd_state_probability(statevector: np.ndarray) -> float:
    """
    Extract probability of measuring ancilla in |1⟩ state.

    For the Hadamard test, measuring the ancilla qubit in |1⟩ vs |0⟩ gives
    information about the expectation value. This function computes the
    probability of measuring |1⟩ by summing over all odd-indexed basis states.

    Args:
        statevector: Quantum statevector from simulation

    Returns:
        Probability of measuring ancilla qubit in |1⟩ state
    """
    probability_sum = 0.0
    for idx in range(len(statevector)):
        if idx % 2 == 1:  # Odd indices correspond to ancilla = |1⟩
            probability = statevector[idx] ** 2
            probability_sum += probability
    return probability_sum


def calculate_vqls_cost(
    parameters: np.ndarray,
    coefficient_set: List[float],
    gate_set: List[List[int]],
    simulator: AerSimulator,
) -> float:
    """
    Calculate the VQLS cost function: C = 1 - |⟨b|A†|ψ⟩|² / ⟨ψ|A†A|ψ⟩.

    This cost function measures how well the variational state |ψ⟩ approximates
    the solution to Ax = b. When C = 0, we have found the exact solution.

    The calculation involves:
    1. Computing ⟨ψ|A†A|ψ⟩ using standard Hadamard tests
    2. Computing ⟨b|A†|ψ⟩ using special Hadamard tests
    3. Combining them into the cost function

    Args:
        parameters: Flattened array of 9 variational parameters
        coefficient_set: Coefficients for the linear combination of unitaries
        gate_set: Gate specifications for building the matrix A
        simulator: Qiskit Aer simulator instance (reused for efficiency)

    Returns:
        Cost function value (0 means perfect solution)
    """
    # Reshape parameters from flat array to 3x3 structure
    param_layers = [parameters[0:3], parameters[3:6], parameters[6:9]]

    # Calculate ⟨ψ|A†A|ψ⟩ term
    overall_sum_aa = 0.0

    for i in range(len(gate_set)):
        for j in range(len(gate_set)):
            # Create circuit for Hadamard test
            circ = QuantumCircuit(NUM_QUBITS)

            # Calculate coefficient for this term
            multiply = coefficient_set[i] * coefficient_set[j]

            # Build Hadamard test circuit
            circ = hadamard_test(
                circ,
                [gate_set[i], gate_set[j]],
                WORK_QUBITS,
                ANCILLA_QUBIT,
                param_layers,
            )

            # Simulate and extract statevector
            circ.save_statevector()
            transpiled_circ = transpile(circ, simulator)
            job = simulator.run(transpiled_circ)
            result = job.result()
            statevector = np.real(result.get_statevector(circ, decimals=100))

            # Extract probability of ancilla = |1⟩
            prob_one = extract_odd_state_probability(statevector)

            # Hadamard test gives: P(1) = (1 - Re[⟨ψ|U|ψ⟩])/2
            # So: Re[⟨ψ|U|ψ⟩] = 1 - 2*P(1)
            expectation = 1 - (2 * prob_one)
            overall_sum_aa += multiply * expectation

    # Calculate |⟨b|A†|ψ⟩|² term
    overall_sum_ab = 0.0

    for i in range(len(gate_set)):
        for j in range(len(gate_set)):
            multiply = coefficient_set[i] * coefficient_set[j]
            product = 1.0

            # Compute ⟨b|A_i†|ψ⟩ and ⟨b|A_j†|ψ⟩
            for gate_idx in range(2):
                circ = QuantumCircuit(NUM_QUBITS)

                # Choose which gate to apply
                selected_gate = gate_set[i] if gate_idx == 0 else gate_set[j]

                # Build special Hadamard test circuit
                circ = special_hadamard_test(
                    circ, selected_gate, WORK_QUBITS, ANCILLA_QUBIT, param_layers, None
                )

                # Simulate and extract statevector
                circ.save_statevector()
                transpiled_circ = transpile(circ, simulator)
                job = simulator.run(transpiled_circ)
                result = job.result()
                statevector = np.real(result.get_statevector(circ, decimals=100))

                # Extract expectation value
                prob_one = extract_odd_state_probability(statevector)
                expectation = 1 - (2 * prob_one)
                product = product * expectation

            overall_sum_ab += multiply * product

    # Calculate final cost function
    cost = 1.0 - float(overall_sum_ab / overall_sum_aa)

    print(f"Cost: {cost:.6f}")

    return cost


def solve_vqls(
    coefficient_set: List[float],
    gate_set: List[List[int]],
    max_iterations: int = 200,
    random_seed: Union[int, None] = None,
) -> OptimizeResult:
    """
    Solve the VQLS problem using variational optimization.

    Args:
        coefficient_set: Coefficients for linear combination of unitaries
        gate_set: Gate specifications for building matrix A
        max_iterations: Maximum number of optimization iterations
        random_seed: Random seed for reproducibility (None for random)

    Returns:
        OptimizeResult object containing optimized parameters and convergence info
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Create simulator once and reuse it
    simulator = AerSimulator()

    # Generate random initial parameters in [0, 3]
    initial_params = [float(random.randint(0, 3000)) / 200 for _ in range(9)]

    # Run optimization
    result = minimize(
        fun=lambda params: calculate_vqls_cost(
            params, coefficient_set, gate_set, simulator
        ),
        x0=initial_params,
        method="COBYLA",
        options={"maxiter": max_iterations},
    )

    return result


def compute_solution_fidelity(
    optimal_parameters: List[List[float]],
    coefficient_set: List[float],
    target_vector: np.ndarray,
) -> float:
    """
    Compute the fidelity between the obtained solution and target vector.

    Args:
        optimal_parameters: Optimized parameters from VQLS
        coefficient_set: Coefficients defining the matrix A
        target_vector: Target vector b (normalized)

    Returns:
        Fidelity (overlap squared) between |ψ⟩ and normalized A|ψ⟩/|b⟩
    """
    # Create circuit with optimal parameters
    circ = QuantumCircuit(3)
    circ = apply_fixed_ansatz(circ, [0, 1, 2], optimal_parameters)
    circ.save_statevector()

    # Simulate to get |ψ⟩
    simulator = AerSimulator()
    transpiled_circ = transpile(circ, simulator)
    job = simulator.run(transpiled_circ)
    result = job.result()
    psi = result.get_statevector(circ, decimals=10)

    # Build matrix A from gate specifications
    # a1 corresponds to Z gate on last qubit (coefficient_set[1])
    a1 = coefficient_set[1] * np.diag([1, 1, 1, 1, -1, -1, -1, -1])

    # a2 corresponds to identity (coefficient_set[0])
    a2 = coefficient_set[0] * np.eye(8)

    # Total matrix A
    matrix_a = a1 + a2

    # Compute A|ψ⟩
    a_psi = matrix_a.dot(psi)

    # Normalize A|ψ⟩
    a_psi_normalized = a_psi / np.linalg.norm(a_psi)

    # Compute fidelity: |⟨b|A|ψ⟩/||A|ψ⟩|||²
    fidelity = np.abs(target_vector.dot(a_psi_normalized)) ** 2

    return fidelity


def main() -> None:
    """
    Main function to run the VQLS algorithm.

    Solves a simple linear system with:
    - Matrix A = 0.55*I + 0.45*Z₃ (where Z₃ acts on the last qubit)
    - Vector b = |+++⟩ (equal superposition state)
    """
    print("=" * 60)
    print("Variational Quantum Linear Solver (VQLS)")
    print("=" * 60)

    # Define the linear system
    coefficient_set = [0.55, 0.225, 0.225]  # Coefficients
    gate_set = [[0, 0, 0], [0, 1, 1]]  # Gate specifications

    # Solve VQLS
    print("\nStarting optimization...")
    result = solve_vqls(coefficient_set, gate_set, max_iterations=1000)

    print("\n" + "=" * 60)
    print("Optimization Results:")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Iterations: {result.nfev}")
    print(f"Final cost: {result.fun:.6f}")

    # Reshape optimal parameters
    optimal_params = [result.x[0:3], result.x[3:6], result.x[6:9]]
    print(f"\nOptimal parameters:")
    for layer_idx, layer in enumerate(optimal_params):
        print(f"  Layer {layer_idx + 1}: {[f'{p:.4f}' for p in layer]}")

    # Compute solution fidelity
    target_b = np.array([1 / np.sqrt(8)] * 8)  # |+++⟩ state
    fidelity = compute_solution_fidelity(optimal_params, coefficient_set, target_b)

    print("\n" + "=" * 60)
    print(f"Solution Fidelity: {fidelity:.6f}")
    print("=" * 60)
    print("\n(Fidelity close to 1.0 indicates successful solution)")


if __name__ == "__main__":
    main()
