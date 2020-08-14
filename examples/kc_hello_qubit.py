"""Creates and simulates a simple circuit.

=== EXAMPLE OUTPUT ===
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
"""

import cirq
import qsimcirq

import qflexcirq.interface.qflex_simulator as qsim
import qflexcirq.interface.qflex_virtual_device as qdevice
import qflexcirq.interface.qflex_grid as qgrid
import qflexcirq.interface.qflex_circuit as qcirc
import qflexcirq.interface.qflex_order as qorder

import qflexcirq.utils as qflexutils

from qflexcirq import qflex

import numpy as np
import math

config_small = {
    'circuit_filename': '/common/users/yh804/research/Google/qflex/config/circuits/rectangular_2x2_1-2-1_0.txt',
    'ordering_filename': '/common/users/yh804/research/Google/qflex/config/ordering/rectangular_2x2.txt',
    'grid_filename': '/common/users/yh804/research/Google/qflex/config/grid/rectangular_2x2.txt',
    'final_state': "0110"
}

def main():

    q0,q1,q2,q3 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0), cirq.GridQubit(1, 1) # cirq.LineQubit.range(4)
    for iteration in range(4):
        random_circuit = cirq.testing.random_circuit(qubits=[q0,q1,q2,q3],
                                                     n_moments=32,
                                                     op_density=0.99)

        print(cirq.qasm(random_circuit))


        my_grid = qgrid.QFlexGrid.from_existing_file(config_small['grid_filename'])
        my_device = qdevice.QFlexVirtualDevice(qflex_grid=my_grid)
        my_order = qorder.QFlexOrder.from_existing_file(config_small["ordering_filename"])


        # circuit_on_device = qcirc.QFlexCircuit(cirq_circuit=random_circuit,
        #                                        device=my_device,
        #                                        qflex_order=my_order,
        #                                        allow_decomposition=True)
        cirq.ConvertToCzAndSingleGates().optimize_circuit(random_circuit) # cannot work with params
        cirq.ExpandComposite().optimize_circuit(random_circuit)


        qs_circuit = qsimcirq.QSimCircuit(random_circuit)




        # The qubits are collected and indexed from the underlying grid_string
        # that was passed as constructor to the Device
        my_qubits = my_device.get_indexed_grid_qubits()

        # Take a QFlex circuit and generate a Cirq circuit from it
        # The Cirq circuit will be afterwards transformed into a Qflex circuit
        # You can construct a Cirq circuit from an existing QFlex circuit
        # Note that circuits provided in files were designed for a specific arrangement
        # my_circuit = qflexutils.GetCircuitOfMoments(config_small["circuit_filename"],
                                                    # my_qubits)



        print("\nRunning QFlex simulation\n")

        my_sim = qsim.QFlexSimulator()
        # myres = my_sim.compute_amplitudes(program=circuit_on_device,
        #                                   bitstrings=[config_small['final_state']])
        # print(myres)

        print("\nRunning Pybind Interface\n")
        print(qflex.simulate(config_small))



        qs_simulator = qsimcirq.QSimSimulator(qsim_options={'t': 16, 'v': 2})
        qs_result = qs_simulator.simulate(qs_circuit)
        assert qs_result.state_vector().shape == (16,)
        kc_simulator = cirq.KnowledgeCompilationSimulator(random_circuit)
        kc_result = kc_simulator.simulate(random_circuit)
        print("qs_result.state_vector()")
        print(qs_result.state_vector())
        print("kc_result.state_vector()")
        print(kc_result.state_vector())
        assert cirq.linalg.allclose_up_to_global_phase(
            qs_result.state_vector(),
            kc_result.state_vector(),
            rtol = 1.e-4,
            atol = 1.e-6,
        )
        circuit_unitary = []
        for x in range(16):
            result = kc_simulator.simulate(random_circuit,
                                        initial_state=x)
            circuit_unitary.append(result.final_state)

        print ("np.transpose(circuit_unitary) = ")
        print (np.transpose(circuit_unitary))
        print ("random_circuit.unitary() = ")
        print (random_circuit.unitary())
        np.testing.assert_almost_equal(
            np.transpose(circuit_unitary),
            random_circuit.unitary(),
            decimal=4)

    # q0,q1 = cirq.LineQubit.range(2)
    # for iteration in range(3):
    #     random_circuit = cirq.testing.random_circuit(qubits=[q0,q1],
    #                                                  n_moments=2,
    #                                                  op_density=0.99)
    #     print("random_circuit:")
    #     print(random_circuit)
    #
    #     # noise = cirq.ConstantQubitNoiseModel(cirq.asymmetric_depolarize(0.2,0.3,0.4)) # mixture: size four noise not implemented
    #     # noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1)) # mixture: size four noise not implemented
    #     # noise = cirq.ConstantQubitNoiseModel(cirq.phase_flip(0.1)) # mixture: works well
    #     # noise = cirq.ConstantQubitNoiseModel(cirq.bit_flip(0.1)) # mixture: works well
    #
    #     noise = cirq.ConstantQubitNoiseModel(cirq.generalized_amplitude_damp(9/25,49/625)) # channel: size four noise not implemented
    #     # noise = cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.1)) # channel:
    #     # noise = cirq.ConstantQubitNoiseModel(cirq.phase_damp(0.1)) # channel:
    #     # reset?
    #
    #     kc_simulator = cirq.KnowledgeCompilationSimulator(random_circuit, noise=noise, intermediate=True)
    #     dm_simulator = cirq.DensityMatrixSimulator(noise=noise)
    #
    #     for initial_state in range(3):
    #
    #         kc_result = kc_simulator.simulate(random_circuit, initial_state=initial_state)
    #         print("kc_result:")
    #         print(kc_result)
    #
    #         dm_result = dm_simulator.simulate(random_circuit,initial_state=initial_state)
    #         print("dm_result:")
    #         print(dm_result)
    #
    #         np.testing.assert_almost_equal(
    #             kc_result.final_density_matrix,
    #             dm_result.final_density_matrix,
    #             decimal=5)

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
