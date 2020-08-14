"""Creates and simulates a simple circuit.

=== EXAMPLE OUTPUT ===
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
"""

import cirq
import numpy as np
import numbers, cmath, sympy

from unittest import mock
import itertools
import random

class MultiHTestGate(cirq.TwoQubitGate):
    def _decompose_(self, qubits):
        return cirq.H.on_each(*qubits)

def main():

    q0, q1 = cirq.LineQubit.range(2)
    circuit_no_meas = cirq.Circuit(
        cirq.H(q0),
        cirq.phase_flip(9/25)(q0),
        cirq.CNOT(q0,q1),
    )

    kc_simulator = cirq.KnowledgeCompilationSimulator(circuit_no_meas, intermediate=True)
    dm_simulator = cirq.DensityMatrixSimulator()

    kc_sim_result = kc_simulator.simulate(circuit_no_meas)
    dm_sim_result = dm_simulator.simulate(circuit_no_meas)

    np.testing.assert_almost_equal(
        kc_sim_result.final_density_matrix,
        dm_sim_result.final_density_matrix,
        decimal=7)



    circuit = cirq.Circuit(
        circuit_no_meas,
        cirq.measure(q0,q1)
    )

    kc_simulator = cirq.KnowledgeCompilationSimulator(circuit, initial_state=0, intermediate=False)
    repetitions = 65536

    print("kc_sim_result.final_density_matrix")
    print(kc_sim_result.final_density_matrix)

    dm_run_result = dm_simulator.run(circuit, repetitions=repetitions)
    dm_histogram = np.zeros(1<<2)
    for bitstring in dm_run_result.measurements['0,1']:
        integer = 0
        for pos, bit in enumerate(bitstring):
            integer += bit<<pos
        dm_histogram[integer] += 1
    print ("DM")
    print (dm_histogram/repetitions)

    kc_run_result = kc_simulator.run(circuit, repetitions=repetitions)
    kc_histogram = np.zeros(1<<2)
    for bitstring in kc_run_result.measurements['0,1']:
        integer = 0
        for pos, bit in enumerate(bitstring):
            integer += bit<<pos
        kc_histogram[integer] += 1
    print ("KC")
    print (kc_histogram/repetitions)

    for _ in range (2):
        dm_run_result = dm_simulator.run(circuit, repetitions=repetitions)
        dm_histogram = np.zeros(1<<2)
        for bitstring in dm_run_result.measurements['0,1']:
            integer = 0
            for pos, bit in enumerate(bitstring):
                integer += bit<<pos
            dm_histogram[integer] += 1
        print ("DM")
        print (dm_histogram/repetitions)

    exit()

if __name__ == '__main__':
    main()
