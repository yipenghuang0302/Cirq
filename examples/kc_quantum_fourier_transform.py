"""
Creates and simulates a circuit for Quantum Fourier Transform(QFT)
on a 4 qubit system.

In this example we demonstrate Fourier Transform on
(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) vector. To do the same, we prepare
the input state of the qubits as |0000>.
=== EXAMPLE OUTPUT ===

Circuit:
(0, 0): ─H───@^0.5───×───H────────────@^0.5─────×───H──────────@^0.5──×─H
             │       │                │         │               │     │
(0, 1): ─────@───────×───@^0.25───×───@─────────×───@^0.25───×──@─────×──
                         │        │                 │        │
(1, 0): ─────────────────┼────────┼───@^0.125───×───┼────────┼───────────
                         │        │   │         │   │        │
(1, 1): ─────────────────@────────×───@─────────×───@────────×───────────

FinalState
[0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j
 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j]
"""

import numpy as np

import cirq
import qsimcirq


def main():
    """Demonstrates Quantum Fourier transform.
    """
    # Create circuit
    qft_circuit = generate_2x2_grid_qft_circuit()
    cirq.ConvertToCzAndSingleGates().optimize_circuit(qft_circuit) # cannot work with params
    cirq.ExpandComposite().optimize_circuit(qft_circuit)

    print('Circuit:')
    print(qft_circuit)
    qsim_circuit = qsimcirq.QSimCircuit(
        cirq_circuit=qft_circuit,
        allow_decomposition=True
    )

    # Simulate and collect final_state
    sv_simulator = cirq.Simulator()
    qs_simulator = qsimcirq.QSimSimulator(qsim_options={'t':16,'v':2})
    sv_result = sv_simulator.simulate(qft_circuit)
    qs_result = qs_simulator.simulate(qsim_circuit)
    assert sv_result.state_vector().shape == (16,)
    assert qs_result.state_vector().shape == (16,)
    assert cirq.linalg.allclose_up_to_global_phase(
        sv_result.state_vector(), qs_result.state_vector())

    kc_simulator = cirq.KnowledgeCompilationSimulator(qft_circuit,intermediate=True)
    kc_result = kc_simulator.simulate(qft_circuit)
    print()
    print('FinalState')
    assert cirq.linalg.allclose_up_to_global_phase(
        sv_result.state_vector(), kc_result.state_vector())
    print(kc_result.state_vector())
    print(np.around(kc_result.final_state, 3))

def _cz_and_swap(q0, q1, rot):
    yield cirq.decompose(cirq.CZ(q0, q1)**rot)
    yield cirq.SWAP(q0,q1)

# Create a quantum fourier transform circuit for 2*2 planar qubit architecture.
# Circuit is adopted from https://arxiv.org/pdf/quant-ph/0402196.pdf
def generate_2x2_grid_qft_circuit():
    # Define a 2*2 square grid of qubits.
    a,b,c,d = cirq.LineQubit.range(4)

    circuit = cirq.Circuit(cirq.H(a),
                           _cz_and_swap(a, b, 0.5),
                           _cz_and_swap(b, c, 0.25),
                           _cz_and_swap(c, d, 0.125),
                           cirq.H(a),
                           _cz_and_swap(a, b, 0.5),
                           _cz_and_swap(b, c, 0.25),
                           cirq.H(a),
                           _cz_and_swap(a, b, 0.5),
                           cirq.H(a),
                           strategy=cirq.InsertStrategy.EARLIEST)
    return circuit

if __name__ == '__main__':
    main()
