"""Creates and simulates a simple circuit.
"""

import cirq
import qsimcirq
import os

import numpy as np

def main():

    q0 = cirq.LineQubit(0)

    circuit = cirq.Circuit( cirq.generalized_amplitude_damp(9/25,49/625)(q0) )
    print("circuit:")
    print(circuit)

    initial_state = 1

    # sv_simulator = cirq.Simulator()
    dm_simulator = cirq.DensityMatrixSimulator()
    kc_simulator = cirq.KnowledgeCompilationSimulator( circuit, initial_state=initial_state  )

    # sv_result = sv_simulator.simulate( circuit, initial_state=initial_state )
    # print("sv_result.state_vector():")
    # print(sv_result.state_vector())

    dm_result = dm_simulator.simulate( circuit, initial_state=initial_state )
    print("dm_result:")
    print(dm_result)

    kc_result = kc_simulator.simulate( circuit )
    print("kc_result:")
    print(kc_result)

    np.testing.assert_almost_equal(
        kc_result.final_density_matrix,
        dm_result.final_density_matrix,
        decimal=4)

if __name__ == '__main__':
    main()
