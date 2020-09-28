"""Runs the Quantum Approximate Optimization Algorithm on Max-Cut.

=== EXAMPLE OUTPUT ===

Example QAOA circuit:
0         1         2
│         │         │
H         H         H
│         │         │
ZZ────────ZZ^0.974  │
│         │         │
Rx(0.51π) ZZ────────ZZ^0.974
│         │         │
│         Rx(0.51π) Rx(0.51π)
│         │         │
M('m')────M─────────M
│         │         │
Optimizing objective function ...
The largest cut value found was 2.0.
The largest possible cut has size 2.0.
The approximation ratio achieved is 1.0.
"""

import itertools

import numpy as np
import networkx
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import cirq


def main(repetitions=1000, maxiter=2):

    # Set problem parameters
    n=3
    p=2

    # Recreate the minimal 3-node example from class
    graph = networkx.Graph()
    graph.add_nodes_from([0,1,2])
    graph.add_edges_from([(0,1), (1,2)])
    networkx.draw(graph)
    plt.show()
    quit()

    # Each node in n nodes of the MAX-CUT graph corresponds to one of n qubits in the quantum circuit.
    # The state vector across the qubits encodes a node partitioning
    qubits = cirq.LineQubit.range(n)

    hadamard_circuit = cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits)
    )
    simulator = cirq.Simulator()
    print ("Put the initial state vector in a superposition of all possible node partitionings:")
    print (simulator.simulate(hadamard_circuit, initial_state=0))
    quit()

    # Print an example circuit
    # Provide classical parameters such that the classical computer can control quantum partitioning
    betas = np.random.uniform(-np.pi, np.pi, size=p)
    gammas = np.random.uniform(-np.pi, np.pi, size=p)
    circuit = qaoa_max_cut_circuit(qubits, betas, gammas, graph)
    print('Example QAOA circuit:')
    print(circuit.to_text_diagram(transpose=True))
    quit()

    # Create variables to store the largest cut and cut value found
    largest_cut_found = None
    largest_cut_value_found = 0

    # Initialize simulator
    simulator = cirq.Simulator()

    # For plotting histograms as a function of the optimization iteration
    histograms = {}
    # Define objective function (we'll use the negative expected cut value)
    optimization_iter = 0
    def f(x):
        # Create circuit
        betas = x[:p]
        gammas = x[p:]
        # Perform a series of operations parameterized by classical parameters
        # such that the final state vector is a superposition of good partitionings
        circuit = qaoa_max_cut_circuit(qubits, betas, gammas, graph)
        circuit_no_measurement = qaoa_max_cut_circuit_no_measurement(qubits, betas, gammas, graph)
        print (simulator.simulate(circuit_no_measurement))
        quit()

        final_state = simulator.simulate(circuit_no_measurement).final_state
        nonlocal optimization_iter
        nonlocal histograms

        histograms[optimization_iter] = np.square(np.absolute(final_state))

        # Sample bitstrings from circuit
        result = simulator.run(circuit, repetitions=repetitions)
        bitstrings = result.measurements['m']
        # Process bitstrings
        nonlocal largest_cut_found
        nonlocal largest_cut_value_found
        values = cut_values(bitstrings, graph)
        max_value_index = np.argmax(values)
        max_value = values[max_value_index]
        if max_value > largest_cut_value_found:
            largest_cut_value_found = max_value
            largest_cut_found = bitstrings[max_value_index]
        mean = np.mean(values)

        optimization_iter += 1
        return -mean

    # Pick an initial guess
    x0 = np.random.uniform(-np.pi, np.pi, size=2 * p)

    # Optimize for a good set of gammas and betas
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
    print('The largest cut value found was {}.'.format(largest_cut_value_found))
    print('The largest possible cut has size {}.'.format(max_cut_value))
    print('The approximation ratio achieved is {}.'.format(
        largest_cut_value_found / max_cut_value))

    # Plot histograms as function of optimization iteration
    fig, ax = plt.subplots()
    basis_states = [i for i in range(1<<n)]
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:03b}"))
    ax.xaxis.set_ticks(np.arange(0, 111, 1))
    print (histograms)
    for opt_iter, histogram in histograms.items():
        plt.plot(basis_states, histogram, label='iter='+str(opt_iter))
    plt.legend(loc='upper right')
    plt.show()


def rzz(rads):
    """Returns a gate with the matrix exp(-i Z⊗Z rads)."""
    return cirq.ZZPowGate(exponent=2 * rads / np.pi, global_shift=-0.5)


def qaoa_max_cut_unitary(qubits, betas, gammas,
                         graph):  # Nodes should be integers
    for beta, gamma in zip(betas, gammas):
        # Need an operator (quantum gate) that encodes an edge
        yield (
            rzz(-0.5 * gamma).on(qubits[i], qubits[j]) for i, j in graph.edges)
        yield cirq.rx(2 * beta).on_each(*qubits)


def qaoa_max_cut_circuit(qubits, betas, gammas,
                         graph):  # Nodes should be integers
    return cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, betas, gammas, graph),
        # Measure
        cirq.measure(*qubits, key='m'))


def qaoa_max_cut_circuit_no_measurement(qubits, betas, gammas,
                         graph):  # Nodes should be integers
    return cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, betas, gammas, graph))


def cut_values(bitstrings, graph):
    mat = networkx.adjacency_matrix(graph, nodelist=sorted(graph.nodes))
    vecs = (-1)**bitstrings
    vals = 0.5 * np.sum(vecs * (mat @ vecs.T).T, axis=-1)
    vals = 0.5 * (graph.size() - vals)
    return vals


if __name__ == '__main__':
    main()
