"""Creates and simulates a simple circuit.

=== EXAMPLE OUTPUT ===
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
"""

import cirq
import qsimcirq
import numpy as np
import math

def main():

    # q0,q1,q2,q3 = cirq.LineQubit.range(4)
    # for iteration in range(4):
    #     random_circuit = cirq.testing.random_circuit(qubits=[q0,q1,q2,q3],
    #                                                  n_moments=32,
    #                                                  op_density=0.99)
    #     cirq.ConvertToCzAndSingleGates().optimize_circuit(random_circuit) # cannot work with params
    #     cirq.ExpandComposite().optimize_circuit(random_circuit)
    #     qs_circuit = qsimcirq.QSimCircuit(random_circuit)
    #     qs_simulator = qsimcirq.QSimSimulator(qsim_options={'t': 16, 'v': 2})
    #     qs_result = qs_simulator.simulate(qs_circuit)
    #     assert qs_result.state_vector().shape == (16,)
    #     kc_simulator = cirq.KnowledgeCompilationSimulator(random_circuit)
    #     kc_result = kc_simulator.simulate(random_circuit)
    #     print("qs_result.state_vector()")
    #     print(qs_result.state_vector())
    #     print("kc_result.state_vector()")
    #     print(kc_result.state_vector())
    #     assert cirq.linalg.allclose_up_to_global_phase(
    #         qs_result.state_vector(),
    #         kc_result.state_vector(),
    #         rtol = 1.e-4,
    #         atol = 1.e-6,
    #     )
    #     circuit_unitary = []
    #     for x in range(16):
    #         result = kc_simulator.simulate(random_circuit,
    #                                     initial_state=x)
    #         circuit_unitary.append(result.final_state)
    #
    #     print ("np.transpose(circuit_unitary) = ")
    #     print (np.transpose(circuit_unitary))
    #     print ("random_circuit.unitary() = ")
    #     print (random_circuit.unitary())
    #     np.testing.assert_almost_equal(
    #         np.transpose(circuit_unitary),
    #         random_circuit.unitary(),
    #         decimal=4)

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

    # q0 = cirq.LineQubit(0)
    #
    # circuit = cirq.Circuit( cirq.generalized_amplitude_damp(9/25,49/625)(q0) )
    # print("circuit:")
    # print(circuit)
    #
    # initial_state = 1
    #
    # # sv_simulator = cirq.Simulator()
    # dm_simulator = cirq.DensityMatrixSimulator()
    # kc_simulator = cirq.KnowledgeCompilationSimulator( circuit, initial_state=initial_state  )
    #
    # # sv_result = sv_simulator.simulate( circuit, initial_state=initial_state )
    # # print("sv_result.state_vector():")
    # # print(sv_result.state_vector())
    #
    # dm_result = dm_simulator.simulate( circuit, initial_state=initial_state )
    # print("dm_result:")
    # print(dm_result)
    #
    # kc_result = kc_simulator.simulate( circuit )
    # print("kc_result:")
    # print(kc_result)
    #
    # np.testing.assert_almost_equal(
    #     kc_result.final_density_matrix,
    #     dm_result.final_density_matrix,
    #     decimal=4)

if __name__ == '__main__':
    main()
