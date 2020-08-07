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
import math
import numpy as np
import sympy
import networkx
import scipy.optimize

import cirq
import time

from collections import defaultdict

def main(
    repetitions=256,
    maxiter=1
    ):
    # Set problem parameters
    kc_kl_divs={}
    dm_kl_divs={}
    for n in range (4,32,2):

        kc_kl_divs[n]=[]
        dm_kl_divs[n]=[]

        for print_n in range(4,n,2):
            for kc_kl_div, dm_kl_div in zip(kc_kl_divs[print_n],dm_kl_divs[print_n]):
                print("print_n="+str(print_n)+" kc_kl_div="+str(kc_kl_div)+" dm_kl_div="+str(dm_kl_div))

        for _ in range(16):

            p = 1

            # Generate a random 3-regular graph on n nodes
            graph = networkx.random_regular_graph(3, n)

            # Make qubits
            qubits = cirq.LineQubit.range(n)

            # Print an example circuit
            betas = np.random.uniform(-np.pi, np.pi, size=p)
            gammas = np.random.uniform(-np.pi, np.pi, size=p)
            circuit_no_meas = qaoa_max_cut_circuit_no_meas(qubits, p, graph)
            print('Example QAOA circuit:')
            print(circuit_no_meas.to_text_diagram(transpose=True))

            # noise = cirq.ConstantQubitNoiseModel(cirq.generalized_amplitude_damp(0.1,0.0))
            # noise = cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.25))
            # noise = cirq.ConstantQubitNoiseModel(cirq.phase_damp(0.25))

            # noise = cirq.ConstantQubitNoiseModel(cirq.asymmetric_depolarize(0.2,0.3,0.4)) # asymmetric depolarizing
            # noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.5)) # symmetric depolarizing
            # noise = cirq.ConstantQubitNoiseModel(cirq.phase_flip(0.25)) # mixture
            # noise = cirq.ConstantQubitNoiseModel(cirq.bit_flip(0.03125)) # mixture

            # circuit_no_meas = cirq.Circuit(cirq.NoiseModel.from_noise_model_like(noise).noisy_moments(circuit_no_meas, sorted(circuit_no_meas.all_qubits())))
            circuit = cirq.Circuit( circuit_no_meas, cirq.measure(*qubits, key='m') )
            cirq.optimizers.ExpandComposite().optimize_circuit(circuit) # seems to actually increase BN size

            # Initialize simulator
            kc_simulator = cirq.KnowledgeCompilationSimulator( circuit, initial_state=0 )
            dm_simulator = cirq.DensityMatrixSimulator()

            # Create variables to store the largest cut and cut value found
            kc_largest_cut_found = None
            kc_largest_cut_value_found = 0
            dm_largest_cut_found = None
            dm_largest_cut_value_found = 0

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
                # kc_sim_result = kc_simulator.simulate(circuit_no_meas, param_resolver=param_resolver)
                dm_sim_result = dm_simulator.simulate(circuit_no_meas, param_resolver=param_resolver)
                # print("kc_sim_result.final_density_matrix")
                # print(kc_sim_result.final_density_matrix)
                # print("dm_sim_result.final_density_matrix")
                # print(dm_sim_result.final_density_matrix)
                # np.testing.assert_almost_equal(
                #     kc_sim_result.final_density_matrix,
                #     dm_sim_result.final_density_matrix,
                #     decimal=4
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
                dm_smp_start = time.time()
                dm_smp_result = dm_simulator.run(circuit, param_resolver=param_resolver, repetitions=repetitions)
                dm_smp_time = time.time() - dm_smp_start
                dm_bitstrings = dm_smp_result.measurements['m']

                # Process histogram
                dm_histogram = defaultdict(int)
                for bitstring in dm_bitstrings:
                    integer = 0
                    for pos, bit in enumerate(bitstring):
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

                nonlocal iter
                # PRINT HISTOGRAMS
                # for index, amplitude in enumerate(dm_sim_result.state_vector()):
                #     bitstring = format(index,'b').zfill(n)
                #     probability = abs(amplitude) * abs(amplitude)
                #     print ('iter='+str(iter)+' bitstring='+str(index)+' kc_samples='+str(kc_histogram[index]/repetitions)+' dm_samples='+str(dm_histogram[index]/repetitions)+' dm_probability='+str(probability))
                kc_kl_div = 0
                dm_kl_div = 0
                for index, probability in enumerate(cirq.sim.density_matrix_utils._probs(
                    dm_sim_result.final_density_matrix,
                    [index for index in range(n)],
                    cirq.protocols.qid_shape(qubits)
                    )):
                    bitstring = format(index,'b').zfill(n)
                    kc_samples = kc_histogram[index]/repetitions
                    dm_samples = dm_histogram[index]/repetitions
                    kc_kl_div += 0 if kc_samples==0 else kc_samples*math.log(kc_samples/probability)
                    dm_kl_div += 0 if dm_samples==0 else dm_samples*math.log(dm_samples/probability)
                    # print ('iter='+str(iter)+' bitstring='+str(index)+' kc_samples='+str(kc_samples)+' dm_samples='+str(dm_samples)+' dm_probability='+str(probability))

                kc_kl_divs[n].append(kc_kl_div)
                dm_kl_divs[n].append(dm_kl_div)

                # print ('n='+str(n)+' kc_kl_div='+str(kc_kl_div)+' dm_kl_div='+str(dm_kl_div))
                # print ('kc_mean='+str(kc_mean)+' dm_mean='+str(dm_mean))
                # print ( 'kc_smp_time=' + str(kc_smp_time) + ' dm_smp_time=' + str(dm_smp_time) )
                # print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                iter += 1
                return -dm_mean

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
