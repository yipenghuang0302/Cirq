"""Runs the Quantum Approximate Optimization Algorithm on Max-Cut.

=== EXAMPLE OUTPUT ===

Example QAOA circuit:
  0           1           2           3           4           5
  │           │           │           │           │           │
  H           H           H           H           H           H
  │           │           │           │           │           │
  ZZ──────────ZZ^(-4/13)  │           │           │           │
┌ │           │           │           │           │           │           ┐
│ ZZ──────────┼───────────ZZ^(-4/13)  │           │           │           │
│ │           ZZ──────────┼───────────ZZ^(-4/13)  │           │           │
└ │           │           │           │           │           │           ┘
┌ │           │           │           │           │           │           ┐
│ ZZ──────────┼───────────┼───────────┼───────────ZZ^(-4/13)  │           │
│ │           ZZ──────────┼───────────┼───────────┼───────────ZZ^(-4/13)  │
└ │           │           │           │           │           │           ┘
  Rx(0.151π)  Rx(0.151π)  ZZ──────────┼───────────ZZ^(-4/13)  │
  │           │           │           │           │           │
  ZZ──────────ZZ^-0.941   ZZ──────────┼───────────┼───────────ZZ^(-4/13)
  │           │           │           ZZ──────────ZZ^(-4/13)  │
┌ │           │           │           │           │           │           ┐
│ │           │           Rx(0.151π)  ZZ──────────┼───────────ZZ^(-4/13)  │
│ │           │           │           │           Rx(0.151π)  │           │
└ │           │           │           │           │           │           ┘
  ZZ──────────┼───────────ZZ^-0.941   Rx(0.151π)  │           Rx(0.151π)
┌ │           │           │           │           │           │           ┐
│ ZZ──────────┼───────────┼───────────┼───────────ZZ^-0.941   │           │
│ │           ZZ──────────┼───────────ZZ^-0.941   │           │           │
└ │           │           │           │           │           │           ┘
  Rx(-0.448π) ZZ──────────┼───────────┼───────────┼───────────ZZ^-0.941
  │           │           ZZ──────────┼───────────ZZ^-0.941   │
  │           │           │           │           │           │
  │           Rx(-0.448π) ZZ──────────┼───────────┼───────────ZZ^-0.941
  │           │           │           ZZ──────────ZZ^-0.941   │
┌ │           │           │           │           │           │           ┐
│ │           │           Rx(-0.448π) ZZ──────────┼───────────ZZ^-0.941   │
│ │           │           │           │           Rx(-0.448π) │           │
└ │           │           │           │           │           │           ┘
  │           │           │           Rx(-0.448π) │           Rx(-0.448π)
  │           │           │           │           │           │
  M('m')──────M───────────M───────────M───────────M───────────M
  │           │           │           │           │           │
Optimizing objective function ...
The largest cut value found was 7.
The largest possible cut has size 7.
The approximation ratio achieved is 1.0.
"""

import itertools

import numpy as np
import sympy
import networkx
import scipy.optimize

import cirq
import qsimcirq
import time

from collections import defaultdict

def main(repetitions=1024, maxiter=50):
    # Set problem parameters
    n = 28
    p = 1

    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(1, n)

    # Make qubits
    qubits = cirq.LineQubit.range(n)

    # Print an example circuit
    betas = np.random.uniform(-np.pi, np.pi, size=p)
    gammas = np.random.uniform(-np.pi, np.pi, size=p)
    circuit_no_meas = qaoa_max_cut_circuit_no_meas(qubits, p, graph)
    print('Example QAOA circuit:')
    print(circuit_no_meas.to_text_diagram(transpose=True))
    circuit = cirq.Circuit( circuit_no_meas, cirq.measure(*qubits, key='m') )

    # Initialize simulator
    kc_simulator = cirq.KnowledgeCompilationSimulator( circuit, initial_state=0 )
    qs_simulator = qsimcirq.QSimSimulator( qsim_options={'t': 16, 'v': 2} )

    # Create variables to store the largest cut and cut value found
    kc_largest_cut_found = None
    kc_largest_cut_value_found = 0
    qs_largest_cut_found = None
    qs_largest_cut_value_found = 0

    # Define objective function (we'll use the negative expected cut value)
    iter = 0
    def f(x):
        # Create circuit
        betas = x[:p]
        betas_dict = { 'beta'+str(index):betas[index] for index in range(p) }
        gammas = x[p:]
        gammas_dict = { 'gamma'+str(index):gammas[index] for index in range(p) }
        param_resolver = cirq.ParamResolver({**betas_dict,**gammas_dict})

        # VALIDATE STATE VECTOR SIMULATION
        solved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        cirq.ConvertToCzAndSingleGates().optimize_circuit(solved_circuit) # cannot work with params
        cirq.ExpandComposite().optimize_circuit(solved_circuit)
        qs_circuit = qsimcirq.QSimCircuit(solved_circuit)

        # kc_sim_result = kc_simulator.simulate(circuit_no_meas, param_resolver=param_resolver)
        qs_sim_result = qs_simulator.simulate(qs_circuit)

        # print("kc_sim_result.state_vector()=")
        # print(kc_sim_result.state_vector())
        # print("qs_sim_result.state_vector()=")
        # print(qs_sim_result.state_vector())

        assert qs_sim_result.state_vector().shape == (1<<n,)
        # assert cirq.linalg.allclose_up_to_global_phase(
        #     kc_sim_result.state_vector(),
        #     qs_sim_result.state_vector(),
        #     rtol = 1.e-5,
        #     atol = 1.e-8,
        # )

        # VALIDATE SAMPLING HISTOGRAMS

        # Sample bitstrings from circuit
        kc_smp_start = time.time()
        kc_smp_result = kc_simulator.run(circuit, param_resolver=param_resolver, repetitions=repetitions)
        kc_smp_time = time.time() - kc_smp_start
        kc_bitstrings = kc_smp_result.measurements['m']

        # Process histogram
        kc_histogram = defaultdict(int)
        for bitstring in kc_bitstrings:
            integer = 0
            for pos, bit in enumerate(bitstring):
                integer += bit<<pos
            kc_histogram[integer] += 1

        # Process bitstrings
        nonlocal kc_largest_cut_found
        nonlocal kc_largest_cut_value_found
        kc_values = cut_values(kc_bitstrings, graph)
        kc_max_value_index = np.argmax(kc_values)
        kc_max_value = kc_values[kc_max_value_index]
        if kc_max_value > kc_largest_cut_value_found:
            kc_largest_cut_value_found = kc_max_value
            kc_largest_cut_found = kc_bitstrings[kc_max_value_index]
        kc_mean = np.mean(kc_values)

        # Sample bitstrings from circuit
        qs_smp_start = time.time()
        qs_smp_result = qs_simulator.run(qs_circuit, repetitions=repetitions)
        qs_smp_time = time.time() - qs_smp_start
        qs_bitstrings = qs_smp_result.measurements['m']

        # Process histogram
        qs_histogram = defaultdict(int)
        for bitstring in qs_bitstrings:
            integer = 0
            for pos, bit in enumerate(bitstring):
                integer += bit<<pos
            qs_histogram[integer] += 1

        # Process bitstrings
        nonlocal qs_largest_cut_found
        nonlocal qs_largest_cut_value_found
        qs_values = cut_values(qs_bitstrings, graph)
        qs_max_value_index = np.argmax(qs_values)
        qs_max_value = qs_values[qs_max_value_index]
        if qs_max_value > qs_largest_cut_value_found:
            qs_largest_cut_value_found = qs_max_value
            qs_largest_cut_found = qs_bitstrings[qs_max_value_index]
        qs_mean = np.mean(qs_values)

        nonlocal iter
        # PRINT HISTOGRAMS
        # for index, amplitude in enumerate(qs_sim_result.state_vector()):
        #     bitstring = format(index,'b').zfill(n)
        #     probability = abs(amplitude) * abs(amplitude)
        #     print ('iter='+str(iter)+' bitstring='+str(index)+' kc_samples='+str(kc_histogram[index]/repetitions)+' qs_samples='+str(qs_histogram[index]/repetitions)+' qs_probability='+str(probability))

        print ('kc_mean='+str(kc_mean)+' qs_mean='+str(qs_mean))
        print ( 'kc_smp_time=' + str(kc_smp_time) + ' qs_smp_time=' + str(qs_smp_time) )
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        iter += 1
        return -qs_mean

    # Pick an initial guess
    x0 = np.random.uniform(-np.pi, np.pi, size=2 * p)

    # Optimize f
    print('Optimizing objective function ...')
    scipy.optimize.minimize(f,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': maxiter})

    # Compute best possible cut value via brute force search
    all_bitstrings = np.array(list(itertools.product(range(2), repeat=n)))
    all_values = cut_values(all_bitstrings, graph)
    max_cut_value = np.max(all_values)

    # Print the results
    print('The largest cut value found was {}.'.format(qs_largest_cut_value_found))
    print('The largest possible cut has size {}.'.format(max_cut_value))
    print('The approximation ratio achieved is {}.'.format(
        qs_largest_cut_value_found / max_cut_value))


def rzz(rads):
    """Returns a gate with the matrix exp(-i Z⊗Z rads)."""
    return cirq.ZZPowGate(exponent=2 * rads / np.pi, global_shift=-0.5)


def qaoa_max_cut_unitary(qubits, p, graph):  # Nodes should be integers
    for index in range(p):
        yield (
            rzz(-0.5 * sympy.Symbol('gamma'+str(index))).on(qubits[i], qubits[j]) for i, j in graph.edges)
        yield cirq.rx(2 * sympy.Symbol('beta'+str(index))).on_each(*qubits)


def qaoa_max_cut_circuit_no_meas(qubits, p, graph):  # Nodes should be integers
    return cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, p, graph),
        # Measure
        # cirq.measure(*qubits, key='m')
        )


def cut_values(bitstrings, graph):
    mat = networkx.adjacency_matrix(graph, nodelist=sorted(graph.nodes))
    vecs = (-1)**bitstrings
    vals = 0.5 * np.sum(vecs * (mat @ vecs.T).T, axis=-1)
    vals = 0.5 * (graph.size() - vals)
    return vals


if __name__ == '__main__':
    main()
