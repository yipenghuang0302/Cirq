"""Demonstrates Deutsch's algorithm.

Deutsch's algorithm is one of the simplest demonstrations of quantum parallelism
and interference. It takes a black-box oracle implementing a Boolean function
f(x), and determines whether f(0) and f(1) have the same parity using just one
query.  This version of Deutsch's algorithm is a simplified and improved version
from Nielsen and Chuang's textbook.

=== REFERENCE ===

https://en.wikipedia.org/wiki/Deutsch–Jozsa_algorithm

Deutsch, David. "Quantum theory, the Church-Turing Principle and the universal
quantum computer." Proc. R. Soc. Lond. A, 400:97, 1985.

=== EXAMPLE OUTPUT ===

Secret function:
f(x) = <0, 1>
Circuit:
0: ───────H───@───H───M('result')───
              │
1: ───X───H───X─────────────────────
Result f(0)⊕f(1):
result=1
"""

import random

import cirq
from cirq import H, X, CNOT, measure
import qsimcirq
import numpy as np


def main():

    for _ in range(16):
        # Choose qubits to use.
        q0, q1 = cirq.LineQubit.range(2)

        # Pick a secret 2-bit function and create a circuit to query the oracle.
        secret_function = [random.randint(0, 1) for _ in range(2)]
        oracle = make_oracle(q0, q1, secret_function)

        # Embed the oracle into a quantum circuit querying it exactly once.
        circuit_no_meas = make_deutsch_circuit(q0, q1, oracle)
        qs_circuit = qsimcirq.QSimCircuit(circuit_no_meas)

        # Simulate the circuit.
        sv_simulator = cirq.Simulator()
        qs_simulator = qsimcirq.QSimSimulator(qsim_options={'t': 1, 'v': 4})

        sv_result = sv_simulator.simulate(circuit_no_meas)
        assert sv_result.state_vector().shape == (4,)
        qs_result = qs_simulator.simulate(qs_circuit)
        assert qs_result.state_vector().shape == (4,)
        assert cirq.linalg.allclose_up_to_global_phase(
            sv_result.state_vector(), qs_result.state_vector())

        circuit = cirq.Circuit( circuit_no_meas, measure(q0, key='result') )
        kc_simulator = cirq.sim.KnowledgeCompilationSimulator(circuit, intermediate=False)
        kc_result = kc_simulator.simulate(circuit)
        assert kc_result.state_vector().shape == (4,)
        np.testing.assert_almost_equal(
            sv_result.state_vector(),
            kc_result.state_vector(),
            decimal=7)

        print('Secret function:\nf(x) = <{}>'.format(
            ', '.join(str(e) for e in secret_function)))
        print('Circuit:')
        print(circuit)
        print('STATE_VECTOR_SIMULATOR: Result of f(0)⊕f(1):')
        sv_result = sv_simulator.run(circuit)
        print(sv_result)
        print('KNOWLEDGE_COMPILATION_SIMULATOR: Result of f(0)⊕f(1):')
        kc_result = kc_simulator.run(circuit)
        print(kc_result)

        assert sv_result==kc_result

def make_oracle(q0, q1, secret_function):
    """ Gates implementing the secret function f(x)."""

    # coverage: ignore
    if secret_function[0]:
        yield [CNOT(q0, q1), X(q1)]

    if secret_function[1]:
        yield CNOT(q0, q1)


def make_deutsch_circuit(q0, q1, oracle):
    c = cirq.Circuit()

    # Initialize qubits.
    c.append([X(q1), H(q1), H(q0)])

    # Query oracle.
    c.append(oracle)

    # Measure in X basis.
    c.append(H(q0))
    return c


if __name__ == '__main__':
    main()
