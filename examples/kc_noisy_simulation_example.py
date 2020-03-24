"""
Creates and simulate a noisy circuit using cirq.ConstantQubitNoiseModel class.
"""
import cirq


def noisy_circuit_demo(amplitude_damp):
    """Demonstrates a noisy circuit simulation.
    """
    # q = cirq.NamedQubit('q')
    q = cirq.LineQubit(0)
    dm_circuit = cirq.Circuit(
        cirq.measure(q, key='initial_state'),
        cirq.X(q),
        cirq.measure(q, key='after_not_gate'),
    )
    kc_circuit = cirq.Circuit(
        cirq.X(q),
        cirq.measure(q, key='after_not_gate'),
    )
    dm_results = cirq.sample(program=dm_circuit,
                          noise=cirq.ConstantQubitNoiseModel(
                              cirq.amplitude_damp(amplitude_damp)),
                          repetitions=100)
    kc_simulator = cirq.KnowledgeCompilationSimulator(
        kc_circuit,
        intermediate=True,
        noise=cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(amplitude_damp))
    )
    kc_results = kc_simulator.run(kc_circuit,repetitions=100)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("ConstantQubitNoiseModel with amplitude damping of rate",
          cirq.amplitude_damp(amplitude_damp))
    print('DENSITY_MATRIX_SIMULATOR: Sampling of initial state of qubit "q":')
    print(dm_results.histogram(key='initial_state'))
    print('DENSITY_MATRIX_SIMULATOR: Sampling of qubit "q" after application of X gate:')
    print(dm_results.histogram(key='after_not_gate'))
    print('KNOWLEDGE_COMPILATION_SIMULATOR: Sampling of qubit "q" after application of X gate:')
    print(kc_results.histogram(key='after_not_gate'))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def main():
    amp_damp_rates = [0, 0.4, 0.5, 1.0]
    for amp_damp_rate in amp_damp_rates:
        noisy_circuit_demo(amp_damp_rate)
        print()


if __name__ == '__main__':
    main()
