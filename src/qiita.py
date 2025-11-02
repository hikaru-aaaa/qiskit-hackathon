import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def apply_fixed_ansatz(qubits, parameters):
    for iz in range(0, len(qubits)):
        circ.ry(parameters[0][iz], qubits[iz])

    circ.cz(qubits[0], qubits[1])
    circ.cz(qubits[2], qubits[0])

    for iz in range(0, len(qubits)):
        circ.ry(parameters[1][iz], qubits[iz])

    circ.cz(qubits[1], qubits[2])
    circ.cz(qubits[2], qubits[0])

    for iz in range(0, len(qubits)):
        circ.ry(parameters[2][iz], qubits[iz])


def control_fixed_ansatz(qubits, parameters, control_qubit, reg):
    for iz in range(0, len(qubits)):
        circ.cry(parameters[0][iz], control_qubit, qubits[iz])

    circ.ccz(control_qubit, qubits[0], qubits[1])
    circ.ccz(control_qubit, qubits[2], qubits[0])

    for iz in range(0, len(qubits)):
        circ.cry(parameters[1][iz], control_qubit, qubits[iz])

    circ.ccz(control_qubit, qubits[1], qubits[2])
    circ.ccz(control_qubit, qubits[2], qubits[0])

    for iz in range(0, len(qubits)):
        circ.cry(parameters[2][iz], control_qubit, qubits[iz])


def had_test(gate_type, qubits, ancilla_index, parameters):
    circ.h(ancilla_index)

    apply_fixed_ansatz(qubits, parameters)

    for ie in range(0, len(gate_type[0])):
        if gate_type[0][ie] == 1:
            circ.cz(ancilla_index, qubits[ie])

    for ie in range(0, len(gate_type[1])):
        if gate_type[1][ie] == 1:
            circ.cz(ancilla_index, qubits[ie])

    circ.h(ancilla_index)


def special_had_test(gate_type, qubits, ancilla_index, parameters, reg):
    circ.h(ancilla_index)

    control_fixed_ansatz(qubits, parameters, ancilla_index, reg)
    for ty in range(0, len(gate_type)):
        if gate_type[ty] == 1:
            circ.cz(ancilla_index, qubits[ty])

    control_b(ancilla_index, qubits)

    circ.h(ancilla_index)


q_reg = QuantumRegister(5)
circ = QuantumCircuit(q_reg)
special_had_test([0, 0, 1], [1, 2, 3], 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]], q_reg)
circ.draw(output="mpl")

circ = QuantumCircuit(4)
had_test([[0, 0, 0], [0, 0, 1]], [1, 2, 3], 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
circ.draw()
circ = QuantumCircuit(3)
apply_fixed_ansatz(
    [0, 1, 2], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
)  # 全て変分パラメーターが1としてセットした例
