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
from random import randrange
import re, os, subprocess

import cirq
import qsimcirq

from statistics import mean, stdev
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

qs_smp_01_time_dict = {}
qs_smp_16_time_dict = {}
qt_smp_01_time_dict = {}
qt_smp_16_time_dict = {}
kc_smp_time_dict = {}

qsim_max = 30

def main(p=1):

    for max_length in range(12,36,2):
        for p in range(1,3):

            vertices_qsim = []
            vertices_all = []

            for n in range(4,max_length,2):

                if n<=qsim_max:
                    vertices_qsim.append(n)
                vertices_all.append(n)

                qs_smp_01_time_dict[n] = []
                qs_smp_16_time_dict[n] = []
                qt_smp_01_time_dict[n] = []
                qt_smp_16_time_dict[n] = []
                kc_smp_time_dict[n] = []

                for _ in range(2):
                    trial(n=n,p=p,repetitions=1000)

            qs_smp_01_time_mean = []
            qs_smp_16_time_mean = []
            qt_smp_01_time_mean = []
            qt_smp_16_time_mean = []
            kc_smp_time_mean = []

            qs_smp_01_time_stdev = []
            qs_smp_16_time_stdev = []
            qt_smp_01_time_stdev = []
            qt_smp_16_time_stdev = []
            kc_smp_time_stdev = []

            for n in vertices_all:

                if n<=qsim_max:
                    qs_smp_01_time_mean.append(mean(qs_smp_01_time_dict[n]))
                    qs_smp_01_time_stdev.append(stdev(qs_smp_01_time_dict[n]))

                    qs_smp_16_time_mean.append(mean(qs_smp_16_time_dict[n]))
                    qs_smp_16_time_stdev.append(stdev(qs_smp_16_time_dict[n]))

                qt_smp_01_time_mean.append(mean(qt_smp_01_time_dict[n]))
                qt_smp_01_time_stdev.append(stdev(qt_smp_01_time_dict[n]))

                qt_smp_16_time_mean.append(mean(qt_smp_16_time_dict[n]))
                qt_smp_16_time_stdev.append(stdev(qt_smp_16_time_dict[n]))

                kc_smp_time_mean.append(mean(kc_smp_time_dict[n]))
                kc_smp_time_stdev.append(stdev(kc_smp_time_dict[n]))

            fig = plt.figure(figsize=(6,3))
            # plt.subplots_adjust(left=.2)
            ax = fig.add_subplot(1, 1, 1)

            ax.set_title('QAOA simulation time vs. qubits (iterations={})'.format(p))
            ax.set_xlabel('Qubits, representing Max-Cut problem vertices')
            ax.set_ylabel('Time (s)')
            ax.set_yscale('log')
            ax.grid(linestyle="--", linewidth=0.25, color='.125', zorder=-10)
            ax.errorbar(vertices_qsim, qs_smp_01_time_mean, yerr=qs_smp_01_time_stdev, color='green' , marker='+', label='qsim sampling with 1 thread')
            ax.errorbar(vertices_qsim, qs_smp_16_time_mean, yerr=qs_smp_16_time_stdev, color='blue', marker='+', label='qsim sampling with 16 threads')
            ax.errorbar(vertices_all, qt_smp_01_time_mean, yerr=qt_smp_01_time_stdev, color='orange' , marker='x', label='qtorch sampling with 1 thread')
            ax.errorbar(vertices_all, qt_smp_16_time_mean, yerr=qt_smp_16_time_stdev, color='red', marker='x', label='qtorch sampling with 16 threads')
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
    meas_circuit = cirq.Circuit( cirq_circuit, cirq.measure(*qubits, key='m') )

    # Initialize simulators
    # sv_sim = cirq.Simulator()
    # dm_simulator = cirq.DensityMatrixSimulator()
    qs_sim_01 = qsimcirq.QSimSimulator( qsim_options={'t': 1, 'v': 0} )
    qs_sim_16 = qsimcirq.QSimSimulator( qsim_options={'t':16, 'v': 0} )
    path_to_qtorch = '/common/home/yh804/research/qtorch/bin/qtorch'
    regex = r"\{(.*?)\}"
    new_circuit_topo = True
    # kc_sim = cirq.KnowledgeCompilationSimulator(cirq_circuit, initial_state=0)
    kc_smp = cirq.KnowledgeCompilationSimulator(meas_circuit, initial_state=0)

    # Create variables to store the largest cut and cut value found
    # sv_largest_cut_found = None
    # sv_largest_cut_value_found = 0
    qs_largest_cut_found = None
    qs_largest_cut_value_found = 0
    kc_largest_cut_found = None
    kc_largest_cut_value_found = 0

    # Define objective function (we'll use the negative expected cut value)
    # eval_iter = 0
    def f(x):
        # Create circuit
        betas = x[:p]
        betas_dict = { 'beta'+str(index):betas[index] for index in range(p) }
        gammas = x[p:]
        gammas_dict = { 'gamma'+str(index):gammas[index] for index in range(p) }
        param_resolver = cirq.ParamResolver({**betas_dict,**gammas_dict})

        # VALIDATE STATE VECTOR SIMULATION
        solved_circuit = cirq.resolve_parameters(meas_circuit, param_resolver)
        solved_circuit._to_quil_output().save_to_file('kc_qtorch_qaoa.quil')

        # sv_sim_start = time.time()
        # sv_sim_result = sv_sim.simulate(cirq_circuit, param_resolver=param_resolver)
        # sv_sim_time = time.time() - sv_sim_start

        # qs_sim_start = time.time()
        # qs_sim_result = qs_sim_16.simulate(cirq_circuit, param_resolver=param_resolver)
        # qs_sim_time = time.time() - qs_sim_start

        # kc_sim_start = time.time()
        # kc_sim_result = kc_sim.simulate(cirq_circuit, param_resolver=param_resolver)
        # kc_sim_time = time.time() - kc_sim_start

        # print("kc_sim_result.state_vector()=")
        # print(kc_sim_result.state_vector())
        # print("qs_sim_result.state_vector()=")
        # print(qs_sim_result.state_vector())

        # assert qs_sim_result.state_vector().shape == (1<<n,)
        # assert cirq.linalg.allclose_up_to_global_phase(
        #     sv_sim_result.state_vector(),
        #     qs_sim_result.state_vector(),
        #     rtol = 1.e-5,
        #     atol = 1.e-7,
        # )
        # assert cirq.linalg.allclose_up_to_global_phase(
        #     qs_sim_result.state_vector(),
        #     kc_sim_result.state_vector(),
        #     rtol = 1.e-4,
        #     atol = 1.e-6,
        # )

        # VALIDATE SAMPLING HISTOGRAMS

        # Sample bitstrings from circuit
        # sv_smp_start = time.time()
        # sv_smp_result = sv_sim.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
        # sv_smp_time = time.time() - sv_smp_start
        # sv_bitstrings = sv_smp_result.measurements['m']

        # Process histogram
        # sv_histogram = defaultdict(int)
        # for bitstring in sv_bitstrings:
        #     integer = 0
        #     for pos, bit in enumerate(reversed(bitstring)):
        #         integer += bit<<pos
        #     sv_histogram[integer] += 1

        # Process bitstrings
        # nonlocal sv_largest_cut_found
        # nonlocal sv_largest_cut_value_found
        # sv_values = cut_values(sv_bitstrings, graph)
        # sv_max_value_index = np.argmax(sv_values)
        # sv_max_value = sv_values[sv_max_value_index]
        # if sv_max_value > sv_largest_cut_value_found:
        #     sv_largest_cut_value_found = sv_max_value
        #     sv_largest_cut_found = sv_bitstrings[sv_max_value_index]
        # sv_mean = np.mean(sv_values)

        # Sample bitstrings from circuit
        if n<=qsim_max:
            qs_smp_01_start = time.time()
            qs_smp_01_result = qs_sim_01.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
            qs_smp_01_time = time.time() - qs_smp_01_start
            qs_smp_01_time_dict[n].append(qs_smp_01_time)

            qs_smp_16_start = time.time()
            qs_smp_16_result = qs_sim_16.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
            qs_smp_16_time = time.time() - qs_smp_16_start
            qs_smp_16_time_dict[n].append(qs_smp_16_time)

            qs_bitstrings = qs_smp_16_result.measurements['m']

            # Process histogram
            qs_histogram = defaultdict(int)
            # qs_bitstring_strs = []
            for bitstring in qs_bitstrings:
                integer = 0
                # string = ''
                for pos, bit in enumerate(bitstring):
                    integer += bit<<pos
                    # string += str(bit)
                qs_histogram[integer] += 1
                # qs_bitstring_strs.append(string)

            # Process bitstrings
            nonlocal qs_largest_cut_found
            nonlocal qs_largest_cut_value_found
            qs_values = cut_values(np.array([np.flip(qs_bitstring) for qs_bitstring in qs_bitstrings]), graph)
            qs_max_value_index = np.argmax(qs_values)
            qs_max_value = qs_values[qs_max_value_index]
            if qs_max_value > qs_largest_cut_value_found:
                qs_largest_cut_value_found = qs_max_value
                qs_largest_cut_found = qs_bitstrings[qs_max_value_index]
            qs_mean = np.mean(qs_values)

#         nonlocal new_circuit_topo
#         if new_circuit_topo:
#             with open('kc_qtorch_qaoa.inp','w') as inp_file:
#                 inp_file.write('''# Line graph decomposition method for contraction
# >string qasm kc_qtorch_qaoa.quil
# # >string contractmethod simple-stoch
# >string contractmethod linegraph-qbb
# >int quickbbseconds 60
# # >string measurement kc_qtorch_qaoa.meas
# # >string outputpath kc_qtorch_qaoa.out
# >bool qbbonly true
# # >bool readqbbresonly true
# ''')
#             stdout = os.system(path_to_qtorch + ' kc_qtorch_qaoa.inp')
#             new_circuit_topo = False

        with open('kc_qtorch_qaoa.inp','w') as inp_file:
            inp_file.write('''# Line graph decomposition method for contraction
>string qasm kc_qtorch_qaoa.quil
>string contractmethod simple-stoch
# >string contractmethod linegraph-qbb
# >int quickbbseconds 60
>string measurement kc_qtorch_qaoa.meas
>string outputpath kc_qtorch_qaoa.out
# >bool qbbonly true
# >bool readqbbresonly true
>int threads 1
''')
        bitstring = randrange(1<<n)
        with open('kc_qtorch_qaoa.meas','w') as meas_file:
            meas_file.write("{:b}".format(bitstring).zfill(n))
        try:
            stdout = subprocess.check_output([path_to_qtorch, 'kc_qtorch_qaoa.inp'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print (stdout)
            print (e.output)
        matches = re.findall(regex, stdout.decode("utf-8"), re.MULTILINE | re.DOTALL)
        begin_time = float(matches[1])
        end_time = float(matches[2])
        qt_smp_01_time_dict[n].append(repetitions*(end_time-begin_time))

        with open('kc_qtorch_qaoa.inp','w') as inp_file:
            inp_file.write('''# Line graph decomposition method for contraction
>string qasm kc_qtorch_qaoa.quil
>string contractmethod simple-stoch
# >string contractmethod linegraph-qbb
# >int quickbbseconds 60
>string measurement kc_qtorch_qaoa.meas
>string outputpath kc_qtorch_qaoa.out
# >bool qbbonly true
# >bool readqbbresonly true
>int threads 16
''')
        bitstring = randrange(1<<n)
        with open('kc_qtorch_qaoa.meas','w') as meas_file:
            meas_file.write("{:b}".format(bitstring).zfill(n))
        stdout = subprocess.check_output([path_to_qtorch, 'kc_qtorch_qaoa.inp'], stderr=subprocess.STDOUT)
        matches = re.findall(regex, stdout.decode("utf-8"), re.MULTILINE | re.DOTALL)
        begin_time = float(matches[1])
        end_time = float(matches[2])
        qt_smp_16_time_dict[n].append(repetitions*(end_time-begin_time))

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

        # nonlocal eval_iter
        # PRINT HISTOGRAMS
        # print ('eval_iter,index,bitstring,bitstring_bin,sv_probability,sv_samples,qs_samples,kc_samples')
        # probabilities = np.zeros(1<<n)
        # for bitstring, amplitude in enumerate(sv_sim_result.state_vector()):
        #     probability = abs(amplitude) * abs(amplitude)
        #     probabilities[bitstring]=probability
        # sorted_bitstrings = np.argsort(probabilities)
        # for index, bitstring in enumerate(sorted_bitstrings):
        #     print (str(eval_iter)+','+str(index)+','+str(bitstring)+','+format(bitstring,'b').zfill(n)+','+str(probabilities[bitstring])+','+"{:.6e}".format(sv_histogram[bitstring]/repetitions)+','+"{:.6e}".format(qs_histogram[bitstring]/repetitions)+','+"{:.6e}".format(kc_histogram[bitstring]/repetitions))

        if n<=qsim_max:
            print ('qs_mean='+str(qs_mean)+' kc_mean='+str(kc_mean))
            # print ( 'sv_sim_time='+str(sv_sim_time)+' qs_sim_time='+str(qs_sim_time)+' kc_sim_time='+str(kc_sim_time) )
            print ( 'qs_smp_01_time='+str(qs_smp_01_time)+' qs_smp_16_time='+str(qs_smp_16_time)+' kc_smp_time='+str(kc_smp_time) )
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # eval_iter += 1
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
    # all_bitstrings = np.array(list(itertools.product(range(2), repeat=n)))
    # all_values = cut_values(all_bitstrings, graph)
    # max_cut_value = np.max(all_values)

    # Print the results
    print('The largest cut value found was {}.'.format(qs_largest_cut_value_found))
    # print('The largest possible cut has size {}.'.format(max_cut_value))
    # print('The approximation ratio achieved is {}.'.format(
    #     qs_largest_cut_value_found / max_cut_value))


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
