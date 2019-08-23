"""Creates and simulates a simple circuit.

=== EXAMPLE OUTPUT ===
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
"""

import cirq
import numpy as np
import math

def main():
    q0, q1, q2 = cirq.LineQubit.range(3)
    for iteration in range(2):
        random_circuit = cirq.testing.random_circuit(qubits=[q0, q1, q2],
                                                     n_moments=8,
                                                     op_density=0.99)
        simulator = cirq.KnowledgeCompilationSimulator(random_circuit)
        circuit_unitary = []
        for x in range(8):
            result = simulator.simulate(random_circuit,
                                        qubit_order=[q0, q1, q2],
                                        initial_state=x)
            circuit_unitary.append(result.final_state)

        print ("np.transpose(circuit_unitary) = ")
        print (np.transpose(circuit_unitary))
        print ("random_circuit.unitary(qubit_order=[q0, q1, q2]) = ")
        print (random_circuit.unitary(qubit_order=[q0, q1, q2]))
        print("iteration=")
        print(iteration)
        np.testing.assert_almost_equal(
            np.transpose(circuit_unitary),
            random_circuit.unitary(qubit_order=[q0, q1, q2]),
            decimal=4)

    # # Pick a qubit.
    # q0, q1 = cirq.LineQubit.range(2)
    #
    # # Create a circuit
    # circuit = cirq.Circuit.from_ops(
    #     cirq.X(qubit)**0.5,  # Square root of NOT.
    #     cirq.measure(qubit, key='m')  # Measurement.
    # )
    # print("Circuit:")
    # print(circuit)
    #
    # # Simulate the circuit several times.
    # simulator = cirq.KnolwedgeCompilationSimulator(circuit)
    # result = simulator.simulate(circuit)
    # print("Results:")
    # print(result)


if __name__ == '__main__':
    main()
