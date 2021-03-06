"""Creates and simulates a circuit equivalent to a Bell inequality test.

=== EXAMPLE OUTPUT ===

Circuit:
(0, 0): ───H───@───X^-0.25───────X^0.5───M───
               │                 │
(0, 1): ───────┼─────────────H───@───────M───
               │
(1, 0): ───────X─────────────────X^0.5───M───
                                 │
(1, 1): ─────────────────────H───@───────M───

Simulating 75 repetitions...

Results
a: _1__11111___111111_1__1_11_11__111________1___1_111_111_____1111_1_111__1_1
b: _1__1__11_1_1_1__1_1_1_1_11____111_111__11_____1_1__1111_1_11___111_11_1__1
x: ____11______11111__1_1111111____1_111__1111___1111_1__11_1__11_11_11_1__11_
y: 11_111_____1_1_111__11111_111_1____1____11_____11___11_1_1___1_111111_1_1_1
(a XOR b) == (x AND y):
   1111_1_111_11111111111111111_1111111__1111_111_111_11111111_11_11111111_111
Win rate: 84.0%
"""

import numpy as np
import cirq
import pytest
import math

def main():
    # Create circuit.
    alice, bob, alice_referee, bob_referee = cirq.LineQubit.range(4)
    circuit = make_bell_test_circuit(alice, bob, alice_referee, bob_referee)

    # Run simulations.
    sv_result = cirq.Simulator().simulate(program=circuit)

    # Then results are recorded.
    circuit.append([
        cirq.measure(alice, key='a'),
        cirq.measure(bob, key='b'),
        cirq.measure(alice_referee, key='x'),
        cirq.measure(bob_referee, key='y'),
    ])
    cirq.optimizers.SynchronizeTerminalMeasurements().optimize_circuit(circuit)
    print('Circuit:')
    print(circuit)
    kc_simulator = cirq.KnowledgeCompilationSimulator(program=circuit, intermediate=False)
    kc_result = kc_simulator.simulate(program=circuit)

    print("sv_result.final_state_vector")
    print(sv_result.final_state_vector)
    print("kc_result.final_state_vector")
    print(kc_result.final_state_vector)

    np.testing.assert_almost_equal(
        sv_result.final_state_vector,
        kc_result.final_state_vector
    )

    repetitions = 7500
    print('Simulating {} repetitions...'.format(repetitions))
    sv_result = cirq.Simulator().run(program=circuit, repetitions=repetitions)
    kc_result = kc_simulator.run(program=circuit, repetitions=repetitions)

    # Collect results.
    a = np.array(kc_result.measurements['a'][:, 0])
    b = np.array(kc_result.measurements['b'][:, 0])
    x = np.array(kc_result.measurements['x'][:, 0])
    y = np.array(kc_result.measurements['y'][:, 0])
    outcomes = a ^ b == x & y
    win_percent = len([e for e in outcomes if e]) * 100 / repetitions

    # Print data.
    print()
    print('Results')
    print('a:', bitstring(a))
    print('b:', bitstring(b))
    print('x:', bitstring(x))
    print('y:', bitstring(y))
    print('(a XOR b) == (x AND y):\n  ', bitstring(outcomes))
    print('Win rate: {}%'.format(win_percent))
    assert win_percent/100. == pytest.approx(math.cos(math.pi/8)*math.cos(math.pi/8),.1)


def make_bell_test_circuit(alice, bob, alice_referee, bob_referee):

    circuit = cirq.Circuit()

    # Prepare shared entangled state.
    circuit.append([
        cirq.H(alice),
        cirq.CNOT(alice, bob),
        cirq.X(alice)**-0.25,
    ])

    # Referees flip coins.
    circuit.append([
        cirq.H(alice_referee),
        cirq.H(bob_referee),
    ])

    # Players do a sqrt(X) based on their referee's coin.
    circuit.append([
        cirq.decompose(cirq.CNOT(alice_referee, alice)**0.5),
        cirq.decompose(cirq.CNOT(bob_referee, bob)**0.5),
    ])

    return circuit


def bitstring(bits):
    return ''.join('1' if e else '_' for e in bits)


if __name__ == '__main__':
    main()
