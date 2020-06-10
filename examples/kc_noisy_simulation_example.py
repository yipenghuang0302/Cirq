"""
Creates and simulate a noisy circuit using cirq.ConstantQubitNoiseModel class.
"""
import cirq
import numpy as np

def noisy_circuit_demo(amplitude_damp):
    """Demonstrates a noisy circuit simulation.
    """
    # q = cirq.NamedQubit('q')
    q = cirq.LineQubit(0)

    dm_circuit = cirq.Circuit(
        cirq.X(q),
    )
    dm_result = cirq.DensityMatrixSimulator(noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(amplitude_damp))).simulate(program=dm_circuit)

    kc_circuit = cirq.Circuit(
        cirq.amplitude_damp(amplitude_damp)(q),
    )
    kc_result = cirq.KnowledgeCompilationSimulator(kc_circuit,initial_state=1,intermediate=True).simulate(kc_circuit)

    print("dm_result.final_density_matrix")
    print(dm_result.final_density_matrix)
    print("kc_result.final_density_matrix")
    print(kc_result.final_density_matrix)

    np.testing.assert_almost_equal(
        dm_result.final_density_matrix,
        kc_result.final_density_matrix
    )

    dm_circuit.append(cirq.measure(q, key='after_not_gate'))
    kc_circuit.append(cirq.measure(q, key='after_not_gate'))

    dm_results = cirq.sample(program=dm_circuit,noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(amplitude_damp)),repetitions=10000)
    kc_simulator = cirq.KnowledgeCompilationSimulator(kc_circuit,initial_state=1,intermediate=False)
    kc_results = kc_simulator.run(kc_circuit,repetitions=10000)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("ConstantQubitNoiseModel with amplitude damping of rate",
          cirq.amplitude_damp(amplitude_damp))
    print('DENSITY_MATRIX_SIMULATOR: Sampling of qubit "q" after application of X gate:')
    print(dm_results.histogram(key='after_not_gate'))
    print('KNOWLEDGE_COMPILATION_SIMULATOR: Sampling of qubit "q" after application of X gate:')
    print(kc_results.histogram(key='after_not_gate'))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def main():
    amp_damp_rates = [0, 9/25, 16/25, 1]
    for amp_damp_rate in amp_damp_rates:
        noisy_circuit_demo(amp_damp_rate)
        print()

if __name__ == '__main__':
    main()
