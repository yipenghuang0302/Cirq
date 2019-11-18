from matplotlib import pyplot as plt

from cirq.contrib.two_qubit_gates import example


def test_gate_compilation_example():
    plt.switch_backend('agg')
    example.main(samples=10, max_infidelity=0.3)
