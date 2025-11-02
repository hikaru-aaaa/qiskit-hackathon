import numpy as np
import random
from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile


def apply_fixed_ansatz(circ, qubits, parameters):
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

    return circ


def control_fixed_ansatz(circ, qubits, parameters, control_qubit, reg):
    for iz in range(0, len(qubits)):
        circ.cry(parameters[0][iz], control_qubit, qubits[iz])

    circ.ccx(control_qubit, qubits[1], 4)
    circ.cz(qubits[0], 4)
    circ.ccx(control_qubit, qubits[1], 4)

    circ.ccx(control_qubit, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(control_qubit, qubits[0], 4)

    for iz in range(0, len(qubits)):
        circ.cry(parameters[1][iz], control_qubit, qubits[iz])

    circ.ccx(control_qubit, qubits[2], 4)
    circ.cz(qubits[1], 4)
    circ.ccx(control_qubit, qubits[2], 4)

    circ.ccx(control_qubit, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(control_qubit, qubits[0], 4)

    for iz in range(0, len(qubits)):
        circ.cry(parameters[2][iz], control_qubit, qubits[iz])

    return circ


def control_b(circ, control_qubit, qubits):
    for ia in qubits:
        circ.ch(control_qubit, ia)
    return circ


def had_test(circ, gate_type, qubits, ancilla_index, parameters):
    circ.h(ancilla_index)

    circ = apply_fixed_ansatz(circ, qubits, parameters)

    for ie in range(0, len(gate_type[0])):
        if gate_type[0][ie] == 1:
            circ.cz(ancilla_index, qubits[ie])

    for ie in range(0, len(gate_type[1])):
        if gate_type[1][ie] == 1:
            circ.cz(ancilla_index, qubits[ie])

    circ.h(ancilla_index)

    return circ


def special_had_test(circ, gate_type, qubits, ancilla_index, parameters, reg):
    circ.h(ancilla_index)

    circ = control_fixed_ansatz(circ, qubits, parameters, ancilla_index, reg)
    for ty in range(0, len(gate_type)):
        if gate_type[ty] == 1:
            circ.cz(ancilla_index, qubits[ty])

    circ = control_b(circ, ancilla_index, qubits)

    circ.h(ancilla_index)

    return circ


def calculate_cost_function(parameters):
    global opt

    overall_sum_1 = 0

    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]

    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):
            qctl = QuantumRegister(5)
            qc = ClassicalRegister(5)
            circ = QuantumCircuit(qctl, qc)

            simulator = AerSimulator()

            multiply = coefficient_set[i] * coefficient_set[j]

            circ = had_test(circ, [gate_set[i], gate_set[j]], [1, 2, 3], 0, parameters)

            circ.save_statevector()
            t_circ = transpile(circ, simulator)
            job = simulator.run(t_circ)

            result = job.result()
            outputstate = np.real(result.get_statevector(circ, decimals=100))
            o = outputstate

            m_sum = 0
            for l in range(0, len(o)):
                if l % 2 == 1:
                    n = o[l] ** 2
                    m_sum += n

            overall_sum_1 += multiply * (1 - (2 * m_sum))

    overall_sum_2 = 0

    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):
            multiply = coefficient_set[i] * coefficient_set[j]
            mult = 1

            for extra in range(0, 2):
                qctl = QuantumRegister(5)
                qc = ClassicalRegister(5)
                circ = QuantumCircuit(qctl, qc)

                simulator = AerSimulator()

                if extra == 0:
                    circ = special_had_test(
                        circ, gate_set[i], [1, 2, 3], 0, parameters, qctl
                    )
                if extra == 1:
                    circ = special_had_test(
                        circ, gate_set[j], [1, 2, 3], 0, parameters, qctl
                    )

                circ.save_statevector()
                t_circ = transpile(circ, simulator)
                job = simulator.run(t_circ)

                result = job.result()
                outputstate = np.real(result.get_statevector(circ, decimals=100))
                o = outputstate

                m_sum = 0
                for l in range(0, len(o)):
                    if l % 2 == 1:
                        n = o[l] ** 2
                        m_sum += n
                mult = mult * (1 - (2 * m_sum))

            overall_sum_2 += multiply * mult

    print(1 - float(overall_sum_2 / overall_sum_1))

    return 1 - float(overall_sum_2 / overall_sum_1)


coefficient_set = [0.55, 0.45]
gate_set = [[0, 0, 0], [0, 0, 1]]

out = minimize(
    calculate_cost_function,
    x0=[float(random.randint(0, 3000)) / 1000 for i in range(0, 9)],
    method="COBYLA",
    options={"maxiter": 200},
)
print(out)

out_f = [out["x"][0:3], out["x"][3:6], out["x"][6:9]]

circ = QuantumCircuit(3, 3)
circ = apply_fixed_ansatz(circ, [0, 1, 2], out_f)
circ.save_statevector()

simulator = AerSimulator()
t_circ = transpile(circ, simulator)
job = simulator.run(t_circ)

result = job.result()
o = result.get_statevector(circ, decimals=10)

a1 = coefficient_set[1] * np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
    ]
)
a2 = coefficient_set[0] * np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)
a3 = np.add(a1, a2)

b = np.array(
    [
        float(1 / np.sqrt(8)),
        float(1 / np.sqrt(8)),
        float(1 / np.sqrt(8)),
        float(1 / np.sqrt(8)),
        float(1 / np.sqrt(8)),
        float(1 / np.sqrt(8)),
        float(1 / np.sqrt(8)),
        float(1 / np.sqrt(8)),
    ]
)

print((b.dot(a3.dot(o) / (np.linalg.norm(a3.dot(o))))) ** 2)
