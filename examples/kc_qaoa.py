# -*- coding: utf-8 -*-
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
import networkx
import sympy
import scipy.optimize

import cirq
import time


def main():
    # Set problem parameters
    n = 6
    p = 2

    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(3, n)

    # Make qubits
    qubits = cirq.LineQubit.range(n)

    # Print an example circuit
    betas = np.random.uniform(-np.pi, np.pi, size=p)
    gammas = np.random.uniform(-np.pi, np.pi, size=p)
    circuit = qaoa_max_cut_circuit(qubits, p, graph)
    print('Example QAOA circuit:')
    print(circuit.to_text_diagram(transpose=True))

    # Create variables to store the largest cut and cut value found
    largest_cut_found = None
    largest_cut_value_found = 0

    # Initialize simulator
    simulator = cirq.KnowledgeCompilationSimulator( circuit, initial_state=0 )

    # Define objective function (we'll use the negative expected cut value)
    def f(x):
        # Create circuit
        betas = x[:p]
        betas_dict = { 'beta'+str(index):betas[index] for index in range(p) }
        gammas = x[p:]
        gammas_dict = { 'gamma'+str(index):gammas[index] for index in range(p) }
        resolver = cirq.ParamResolver({**betas_dict,**gammas_dict})
        # Sample bitstrings from circuit
        start = time.time()
        result = simulator.simulate(
            program=circuit,
            param_resolver=resolver)
        print("simulation time = ")
        print(time.time() - start)
        # Process bitstrings
        mean = 0
        nonlocal largest_cut_found
        nonlocal largest_cut_value_found
        for index, amplitude in enumerate(result._final_simulator_state.state_vector):
            bitstring = format(index,'b').zfill(n)
            value = cut_value(bitstring, graph)
            probability = abs(amplitude) * abs(amplitude)
            mean += value * probability
            if value > largest_cut_value_found:
                largest_cut_value_found = value
                largest_cut_found = bitstring
        print ('mean =')
        print (mean)
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return -mean

    # Pick an initial guess
    x0 = np.random.uniform(-np.pi, np.pi, size=2 * p)

    # Optimize f
    print('Optimizing objective function ...')
    scipy.optimize.minimize(f,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': 50})

    # Compute best possible cut value via brute force search
    max_cut_value = max(
        cut_value(bitstring, graph)
        for bitstring in itertools.product(range(2), repeat=n))

    # Print the results
    print('The largest cut value found was {}.'.format(largest_cut_value_found))
    print('The largest possible cut has size {}.'.format(max_cut_value))
    print('The approximation ratio achieved is {}.'.format(
        largest_cut_value_found / max_cut_value))


def Rzz(rads):
    """Returns a gate with the matrix exp(-i Z⊗Z rads)."""
    return cirq.ZZPowGate(exponent=2 * rads / np.pi, global_shift=-0.5)


def qaoa_max_cut_unitary(qubits, p, graph):  # Nodes should be integers
    for index in range(p):
        yield (
            Rzz(-0.5 * sympy.Symbol('gamma'+str(index))).on(qubits[i], qubits[j]) for i, j in graph.edges)
        yield cirq.Rx(2 * sympy.Symbol('beta'+str(index))).on_each(*qubits)


def qaoa_max_cut_circuit(qubits, p, graph):  # Nodes should be integers
    return cirq.Circuit.from_ops(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, p, graph),
        # Measure
        # cirq.measure(*qubits, key='m')
        )


def cut_value(bitstring, graph):
    return sum(bitstring[i] != bitstring[j] for i, j in graph.edges)


if __name__ == '__main__':
    main()
