"""Creates and simulates a simple circuit.
"""

import cirq
import numpy as np

def main():

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))
    simulator = cirq.KnowledgeCompilationSimulator(circuit, dtype=np.complex64)
    for _ in range(10):
        result = simulator.run(circuit)
        bits = result.measurements['0,1'][0]
        assert bits[0] == bits[1]

if __name__ == '__main__':
    main()
