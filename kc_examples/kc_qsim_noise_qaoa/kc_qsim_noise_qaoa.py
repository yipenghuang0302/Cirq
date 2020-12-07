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
from collections import defaultdict
import time

import networkx
import numpy as np
import sympy
import scipy.optimize

import cirq

from statistics import mean, stdev
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

dm_smp_time_dict = {}
kc_smp_time_dict = {}

cirq_max = 12

def main():

    for p in range(1,3):
        for max_length in range(8,16,2): #36

            vertices_cirq = []
            vertices_all = []
            for n in range(4,max_length,2):

                if n<=cirq_max:
                    vertices_cirq.append(n)
                vertices_all.append(n)

                dm_smp_time_dict[n] = []
                kc_smp_time_dict[n] = []

                for _ in range(2):
                    trial(n=n,p=p,repetitions=1000)

            dm_smp_time_mean = []
            kc_smp_time_mean = []

            dm_smp_time_stdev = []
            kc_smp_time_stdev = []

            for n in vertices_all:

                if n<=cirq_max:
                    dm_smp_time_mean.append(mean(dm_smp_time_dict[n]))
                    dm_smp_time_stdev.append(stdev(dm_smp_time_dict[n]))

                kc_smp_time_mean.append(mean(kc_smp_time_dict[n]))
                kc_smp_time_stdev.append(stdev(kc_smp_time_dict[n]))

            fig = plt.figure(figsize=(6,3))
            # plt.subplots_adjust(left=.2)
            ax = fig.add_subplot(1, 1, 1)

            ax.set_title('Noisy QAOA simulation time vs. qubits (iterations={})'.format(p))
            ax.set_xlabel('Qubits, representing Max-Cut problem vertices')
            ax.set_ylabel('Time (s)')
            ax.set_yscale('log')
            ax.grid(linestyle="--", linewidth=0.25, color='.125', zorder=-10)
            ax.errorbar(vertices_cirq, dm_smp_time_mean, yerr=dm_smp_time_stdev, color='gray', marker='2', label='density matrix sampling')
            ax.errorbar(vertices_all, kc_smp_time_mean, yerr=kc_smp_time_stdev, color='purple', marker='o', label='knowledge compilation sampling')
            ax.legend(loc='upper left', frameon=False)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)

            timestr = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(fname=timestr+'.pdf', format='pdf')

# Set problem parameters
def trial(n=6, p=2, repetitions=1000, maxiter=2):

    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(3, n)

    # Make qubits
    qubits = cirq.LineQubit.range(n)

    # Print an example circuit
    cirq_circuit = qaoa_max_cut_circuit_no_meas(qubits, p, graph)
    print('Example QAOA circuit:')
    print(cirq_circuit.to_text_diagram(transpose=True))

    # noise = cirq.ConstantQubitNoiseModel(cirq.asymmetric_depolarize(0.005,0.005,0.005)) # mixture: size four noise not implemented
    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.005)) # mixture: size four noise not implemented
    # noise = cirq.ConstantQubitNoiseModel(cirq.phase_flip(0.01)) # mixture: works well
    # noise = cirq.ConstantQubitNoiseModel(cirq.bit_flip(0.01)) # mixture: works well

    cirq_circuit = cirq.Circuit(cirq.NoiseModel.from_noise_model_like(noise).noisy_moments(cirq_circuit, sorted(cirq_circuit.all_qubits())))
    meas_circuit = cirq.Circuit( cirq_circuit, cirq.measure(*qubits, key='m') )

    # Initialize simulators
    dm_sim = cirq.DensityMatrixSimulator()
    # kc_sim = cirq.KnowledgeCompilationSimulator(cirq_circuit, initial_state=0)
    kc_smp = cirq.KnowledgeCompilationSimulator(meas_circuit, initial_state=0)

    # Create variables to store the largest cut and cut value found
    dm_largest_cut_found = None
    dm_largest_cut_value_found = 0
    kc_largest_cut_found = None
    kc_largest_cut_value_found = 0

    # Define objective function (we'll use the negative expected cut value)
    iter = 0
    def f(x):
        # Create circuit
        betas = x[:p]
        betas_dict = { 'beta'+str(index):betas[index] for index in range(p) }
        gammas = x[p:]
        gammas_dict = { 'gamma'+str(index):gammas[index] for index in range(p) }
        param_resolver = cirq.ParamResolver({**betas_dict,**gammas_dict})

        # VALIDATE DENSITY MATRIX SIMULATION

        # dm_sim_start = time.time()
        # dm_sim_result = dm_sim.simulate(cirq_circuit, param_resolver=param_resolver)
        # dm_sim_time = time.time() - dm_sim_start

        # kc_sim_start = time.time()
        # kc_sim_result = kc_sim.simulate(cirq_circuit, param_resolver=param_resolver)
        # kc_sim_time = time.time() - kc_sim_start

        # print("dm_sim_result.final_density_matrix=")
        # print(dm_sim_result.final_density_matrix)
        # print("kc_sim_result.final_density_matrix=")
        # print(kc_sim_result.final_density_matrix)
        #
        # np.testing.assert_almost_equal(
        #     dm_sim_result.final_density_matrix,
        #     kc_sim_result.final_density_matrix,
        #     decimal=6
        # )

        # VALIDATE SAMPLING HISTOGRAMS

        # Sample bitstrings from circuit
        if n<=cirq_max:
            dm_smp_start = time.time()
            dm_smp_result = dm_sim.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
            dm_smp_time = time.time() - dm_smp_start
            dm_smp_time_dict[n].append(dm_smp_time)

            # Process histogram
            dm_bitstrings = dm_smp_result.measurements['m']
            dm_histogram = defaultdict(int)
            for bitstring in dm_bitstrings:
                integer = 0
                for pos, bit in enumerate(reversed(bitstring)):
                    integer += bit<<pos
                dm_histogram[integer] += 1

            # Process bitstrings
            nonlocal dm_largest_cut_found
            nonlocal dm_largest_cut_value_found
            dm_values = cut_values(dm_bitstrings, graph)
            dm_max_value_index = np.argmax(dm_values)
            dm_max_value = dm_values[dm_max_value_index]
            if dm_max_value > dm_largest_cut_value_found:
                dm_largest_cut_value_found = dm_max_value
                dm_largest_cut_found = dm_bitstrings[dm_max_value_index]
            dm_mean = np.mean(dm_values)

        # Sample bitstrings from circuit
        kc_smp_start = time.time()
        kc_smp_result = kc_smp.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
        kc_smp_time = time.time() - kc_smp_start
        kc_smp_time_dict[n].append(kc_smp_time)

        # Process histogram
        kc_bitstrings = kc_smp_result.measurements['m']
        kc_histogram = defaultdict(int)
        for bitstring in kc_bitstrings:
            integer = 0
            for pos, bit in enumerate(reversed(bitstring)):
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

        nonlocal iter
        # PRINT HISTOGRAMS
        # print ('iter,index,bitstring,bitstring_bin,dm_probability,dm_samples,kc_samples')
        # probabilities = np.zeros(1<<n)
        # for bitstring, probability in enumerate(cirq.sim.density_matrix_utils._probs(
        #     dm_sim_result.final_density_matrix,
        #     [index for index in range(n)],
        #     cirq.protocols.qid_shape(qubits)
        # )):
        #     probabilities[bitstring]=probability
        # sorted_bitstrings = np.argsort(probabilities)
        # for index, bitstring in enumerate(sorted_bitstrings):
        #     print (str(iter)+','+str(index)+','+str(bitstring)+','+format(bitstring,'b').zfill(n)+','+str(probabilities[bitstring])+','+"{:.6e}".format(dm_histogram[bitstring]/repetitions)+','+"{:.6e}".format(kc_histogram[bitstring]/repetitions))

        if n<=cirq_max:
            print ('dm_mean='+str(dm_mean)+' kc_mean='+str(kc_mean))
            # print ( 'dm_sim_time='+str(dm_sim_time)+' kc_sim_time='+str(kc_sim_time) )
            # print ( 'dm_sim_time='+str(dm_sim_time) )
            print ( 'dm_smp_time='+str(dm_smp_time)+' kc_smp_time='+str(kc_smp_time) )
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        iter += 1
        return -kc_mean

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
    print('The largest cut value found was {}.'.format(dm_largest_cut_value_found))
    print('The largest possible cut has size {}.'.format(max_cut_value))
    print('The approximation ratio achieved is {}.'.format(
        dm_largest_cut_value_found / max_cut_value))


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
