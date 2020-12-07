"""Creates and simulates a simple circuit.
"""

import cirq
import qsimcirq
import os

import numpy as np

def main():

    q0,q1 = cirq.LineQubit.range(2)
    for iteration in range(3):
        random_circuit = cirq.testing.random_circuit(qubits=[q0,q1],
                                                     n_moments=2,
                                                     op_density=0.99)
        print("random_circuit:")
        print(random_circuit)

        # noise = cirq.ConstantQubitNoiseModel(cirq.asymmetric_depolarize(0.2,0.3,0.4)) # mixture: size four noise not implemented
        # noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1)) # mixture: size four noise not implemented
        # noise = cirq.ConstantQubitNoiseModel(cirq.phase_flip(0.1)) # mixture: works well
        # noise = cirq.ConstantQubitNoiseModel(cirq.bit_flip(0.1)) # mixture: works well

        noise = cirq.ConstantQubitNoiseModel(cirq.generalized_amplitude_damp(9/25,49/625)) # channel: size four noise not implemented
        # noise = cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1)) # channel:
        # noise = cirq.ConstantQubitNoiseModel(cirq.phase_damp(0.1)) # channel:
        # reset?

        kc_simulator = cirq.KnowledgeCompilationSimulator(random_circuit, noise=noise, intermediate=True)
        dm_simulator = cirq.DensityMatrixSimulator(noise=noise)

        for initial_state in range(3):

            kc_result = kc_simulator.simulate(random_circuit, initial_state=initial_state)
            print("kc_result:")
            print(kc_result)

            dm_result = dm_simulator.simulate(random_circuit,initial_state=initial_state)
            print("dm_result:")
            print(dm_result)

            np.testing.assert_almost_equal(
                kc_result.final_density_matrix,
                dm_result.final_density_matrix,
                decimal=5)

if __name__ == '__main__':
    main()
