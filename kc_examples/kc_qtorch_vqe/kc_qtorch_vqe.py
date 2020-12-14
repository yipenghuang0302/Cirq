from collections import defaultdict
import time

import random
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
qtorch_max = 25

def main():

    for steps in range(1,2):
        for max_length in range(6,7):

            grid_points_qsim = []
            grid_points_qtorch = []
            grid_points_all = []
            for length in range(2,max_length):

                if length*length<=qsim_max:
                    grid_points_qsim.append(length*length)
                if length*length<=qtorch_max:
                    grid_points_qtorch.append(length*length)
                grid_points_all.append(length*length)

                qs_smp_01_time_dict[length*length] = []
                qs_smp_16_time_dict[length*length] = []
                qt_smp_01_time_dict[length*length] = []
                qt_smp_16_time_dict[length*length] = []
                kc_smp_time_dict[length*length] = []

                for _ in range(2):
                    trial(length=length,steps=steps,repetitions=1000)

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

            for grid_point in grid_points_all:

                if grid_point<=qsim_max:
                    qs_smp_01_time_mean.append(mean(qs_smp_01_time_dict[grid_point]))
                    qs_smp_01_time_stdev.append(stdev(qs_smp_01_time_dict[grid_point]))

                    qs_smp_16_time_mean.append(mean(qs_smp_16_time_dict[grid_point]))
                    qs_smp_16_time_stdev.append(stdev(qs_smp_16_time_dict[grid_point]))

                if grid_point<=qtorch_max:
                    qt_smp_01_time_mean.append(mean(qt_smp_01_time_dict[grid_point]))
                    qt_smp_01_time_stdev.append(stdev(qt_smp_01_time_dict[grid_point]))

                    qt_smp_16_time_mean.append(mean(qt_smp_16_time_dict[grid_point]))
                    qt_smp_16_time_stdev.append(stdev(qt_smp_16_time_dict[grid_point]))

                kc_smp_time_mean.append(mean(kc_smp_time_dict[grid_point]))
                kc_smp_time_stdev.append(stdev(kc_smp_time_dict[grid_point]))

            fig = plt.figure(figsize=(6,3))
            # plt.subplots_adjust(left=.2)
            ax = fig.add_subplot(1, 1, 1)

            ax.set_title('VQE simulation time vs. qubits (iterations={})'.format(steps))
            ax.set_xlabel('Qubits, representing 2D Ising model grid points')
            ax.set_ylabel('Time (s)')
            ax.set_yscale('log')
            ax.grid(linestyle="--", linewidth=0.25, color='.125', zorder=-10)

            ax.errorbar(grid_points_qsim, qs_smp_01_time_mean, yerr=qs_smp_01_time_stdev, color='green' , marker='+', label='qsim sampling with 1 thread')
            ax.errorbar(grid_points_qsim, qs_smp_16_time_mean, yerr=qs_smp_16_time_stdev, color='blue', marker='+', label='qsim sampling with 16 threads')
            ax.errorbar(grid_points_qtorch, qt_smp_01_time_mean, yerr=qt_smp_01_time_stdev, color='orange' , marker='x', label='qtorch sampling with 1 thread')
            ax.errorbar(grid_points_qtorch, qt_smp_16_time_mean, yerr=qt_smp_16_time_stdev, color='red', marker='x', label='qtorch sampling with 16 threads')
            ax.errorbar(grid_points_all, kc_smp_time_mean, yerr=kc_smp_time_stdev, color='purple', marker='o', label='knowledge compilation sampling')
            ax.legend(loc='upper left', frameon=False)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)

            timestr = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(fname=timestr+'.pdf', format='pdf')

# define the length of the grid.
def trial(length=2, steps=2, repetitions=1000, maxiter=2):

    h, jr, jc = random_instance(length)
    print('transverse fields: {}'.format(h))
    print('row j fields: {}'.format(jr))
    print('column j fields: {}'.format(jc))
    # prints something like
    # transverse fields: [[-1, 1, -1], [1, -1, -1], [-1, 1, -1]]
    # row j fields: [[1, 1, -1], [1, -1, 1]]
    # column j fields: [[1, -1], [-1, 1], [-1, 1]]

    # define qubits on the grid.
    # [cirq.GridQubit(i, j) for i in range(length) for j in range(length)]
    qubits = cirq.LineQubit.range(length*length)
    print(qubits)
    # prints
    # [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2), cirq.GridQubit(1, 0), cirq.GridQubit(1, 1), cirq.GridQubit(1, 2), cirq.GridQubit(2, 0), cirq.GridQubit(2, 1), cirq.GridQubit(2, 2)]

    cirq_circuit = cirq.Circuit()
    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')
    gamma = sympy.Symbol('gamma')
    for _ in range(steps):
        cirq_circuit.append(one_step(h, jr, jc, alpha, beta, gamma))
    meas_circuit = cirq.Circuit( cirq_circuit, cirq.measure(*qubits, key='x') )
    print(meas_circuit)

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


    # eval_iter = 0
    def f(x):
        param_resolver = cirq.ParamResolver({ 'alpha':x[0], 'beta':x[1], 'gamma':x[2] })

        # VALIDATE STATE VECTOR SIMULATION
        solved_circuit = cirq.resolve_parameters(meas_circuit, param_resolver)
        solved_circuit._to_quil_output().save_to_file('kc_qtorch_vqe.quil')

        # sv_sim_start = time.time()
        # sv_sim_result = sv_sim.simulate(cirq_circuit, param_resolver=param_resolver)
        # sv_sim_time = time.time() - sv_sim_start
        #
        # qs_sim_start = time.time()
        # qs_sim_result = qs_sim_16.simulate(cirq_circuit, param_resolver=param_resolver)
        # qs_sim_time = time.time() - qs_sim_start
        #
        # kc_sim_start = time.time()
        # kc_sim_result = kc_sim.simulate(cirq_circuit, param_resolver=param_resolver)
        # kc_sim_time = time.time() - kc_sim_start
        #
        # print("kc_sim_result.state_vector()=")
        # print(kc_sim_result.state_vector())
        # print("qs_sim_result.state_vector()=")
        # print(qs_sim_result.state_vector())
        #
        # assert qs_sim_result.state_vector().shape == (1<<(length*length),)
        # assert cirq.linalg.allclose_up_to_global_phase(
        #     sv_sim_result.state_vector(),
        #     qs_sim_result.state_vector(),
        #     rtol = 1.e-4,
        #     atol = 1.e-6,
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
        # sv_bitstrings = sv_smp_result.measurements['x']

        # Process histogram
        # sv_histogram = defaultdict(int)
        # for bitstring in sv_bitstrings:
        #     integer = 0
        #     for pos, bit in enumerate(reversed(bitstring)):
        #         integer += bit<<pos
        #     sv_histogram[integer] += 1

        # Process bitstrings
        # sv_value = obj_func(sv_smp_result, h, jr, jc)
        # print('Objective value is {}.'.format(sv_value))


        # Sample bitstrings from circuit
        if length*length<=qsim_max:
            qs_smp_01_start = time.time()
            qs_smp_01_result = qs_sim_01.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
            qs_smp_01_time = time.time() - qs_smp_01_start
            qs_smp_01_time_dict[length*length].append(qs_smp_01_time)

            qs_smp_16_start = time.time()
            qs_smp_16_result = qs_sim_16.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
            qs_smp_16_time = time.time() - qs_smp_16_start
            qs_smp_16_time_dict[length*length].append(qs_smp_16_time)

            qs_bitstrings = qs_smp_16_result.measurements['x']

            # Process histogram
            qs_histogram = defaultdict(int)
            for bitstring in qs_bitstrings:
                integer = 0
                for pos, bit in enumerate(bitstring):
                    integer += bit<<pos
                qs_histogram[integer] += 1

            # Process bitstrings
            qs_value = obj_func(qs_smp_16_result, h, jr, jc)
            print('Objective value is {}.'.format(qs_value))

        if length*length<=qtorch_max:
            nonlocal new_circuit_topo
            if new_circuit_topo:
                with open('kc_qtorch_vqe.inp','w') as inp_file:
                    inp_file.write('''# Line graph decomposition method for contraction
>string qasm kc_qtorch_vqe.quil
# >string contractmethod simple-stoch
>string contractmethod linegraph-qbb
>int quickbbseconds 300
# >string measurement kc_qtorch_vqe.meas
# >string outputpath kc_qtorch_vqe.out
>bool qbbonly true
# >bool readqbbresonly true
''')
                stdout = os.system(path_to_qtorch + ' kc_qtorch_vqe.inp')
                new_circuit_topo = False

            with open('kc_qtorch_vqe.inp','w') as inp_file:
                inp_file.write('''# Line graph decomposition method for contraction
>string qasm kc_qtorch_vqe.quil
# >string contractmethod simple-stoch
>string contractmethod linegraph-qbb
# >int quickbbseconds 300
>string measurement kc_qtorch_vqe.meas
>string outputpath kc_qtorch_vqe.out
# >bool qbbonly true
>bool readqbbresonly true
>int threads 1
''')
            bitstring = randrange(1<<(length*length))
            with open('kc_qtorch_vqe.meas','w') as meas_file:
                meas_file.write("{:b}".format(bitstring).zfill(length*length))
            try:
                stdout = subprocess.check_output([path_to_qtorch, 'kc_qtorch_vqe.inp'], stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print (stdout)
                print (e.output)
            matches = re.findall(regex, stdout.decode("utf-8"), re.MULTILINE | re.DOTALL)
            begin_time = float(matches[1])
            end_time = float(matches[2])
            qt_smp_01_time_dict[length*length].append(repetitions*(end_time-begin_time))

            with open('kc_qtorch_vqe.inp','w') as inp_file:
                inp_file.write('''# Line graph decomposition method for contraction
>string qasm kc_qtorch_vqe.quil
# >string contractmethod simple-stoch
>string contractmethod linegraph-qbb
# >int quickbbseconds 300
>string measurement kc_qtorch_vqe.meas
>string outputpath kc_qtorch_vqe.out
# >bool qbbonly true
>bool readqbbresonly true
>int threads 16
''')
            bitstring = randrange(1<<(length*length))
            with open('kc_qtorch_vqe.meas','w') as meas_file:
                meas_file.write("{:b}".format(bitstring).zfill(length*length))
            stdout = subprocess.check_output([path_to_qtorch, 'kc_qtorch_vqe.inp'], stderr=subprocess.STDOUT)
            matches = re.findall(regex, stdout.decode("utf-8"), re.MULTILINE | re.DOTALL)
            begin_time = float(matches[1])
            end_time = float(matches[2])
            qt_smp_16_time_dict[length*length].append(repetitions*(end_time-begin_time))

        # Sample bitstrings from circuit
        kc_smp_start = time.time()
        kc_smp_result = kc_smp.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
        kc_smp_time = time.time() - kc_smp_start
        kc_smp_time_dict[length*length].append(kc_smp_time)

        # Process histogram
        kc_bitstrings = kc_smp_result.measurements['x']
        kc_histogram = defaultdict(int)
        for bitstring in kc_bitstrings:
            integer = 0
            for pos, bit in enumerate(reversed(bitstring)):
                integer += bit<<pos
            kc_histogram[integer] += 1

        # Process bitstrings
        kc_value = obj_func(kc_smp_result, h, jr, jc)
        print('Objective value is {}.'.format(kc_value))

        # nonlocal eval_iter
        # PRINT HISTOGRAMS
        # print ('eval_iter,index,bitstring,bitstring_bin,sv_probability,sv_samples,qs_samples,kc_samples')
        # probabilities = np.zeros(1<<(length*length))
        # for bitstring, amplitude in enumerate(sv_sim_result.state_vector()):
        #     probability = abs(amplitude) * abs(amplitude)
        #     probabilities[bitstring]=probability
        # sorted_bitstrings = np.argsort(probabilities)
        # for index, bitstring in enumerate(sorted_bitstrings):
        #     print (str(eval_iter)+','+str(index)+','+str(bitstring)+','+format(bitstring,'b').zfill(length*length)+','+str(probabilities[bitstring])+','+"{:.6e}".format(sv_histogram[bitstring]/repetitions)+','+"{:.6e}".format(qs_histogram[bitstring]/repetitions)+','+"{:.6e}".format(kc_histogram[bitstring]/repetitions))

        if length*length<=qsim_max:
            print ('qs_value='+str(qs_value)+' kc_value='+str(kc_value))
            # print ( 'sv_sim_time='+str(sv_sim_time)+' qs_sim_time='+str(qs_sim_time)+' kc_sim_time='+str(kc_sim_time) )
            print ( 'qs_smp_01_time='+str(qs_smp_01_time)+' qs_smp_16_time='+str(qs_smp_16_time)+' kc_smp_time='+str(kc_smp_time) )
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # eval_iter += 1
        return kc_value

    # Pick an initial guess
    x0 = np.random.uniform(0.0, 1.0, size=3)

    # Optimize f
    print('Optimizing objective function ...')
    scipy.optimize.minimize(f,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': maxiter})


def rand2d(rows, cols):
    return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]

def random_instance(length):
    # transverse field terms
    h = rand2d(length, length)
    # links within a row
    jr = rand2d(length - 1, length)
    # links within a column
    jc = rand2d(length, length - 1)
    return (h, jr, jc)

def energy_func(length, h, jr, jc):
    def energy(measurements):
        # Reshape measurement into array that matches grid shape.
        # meas_list_of_lists = [measurements[i * length:(i + 1) * length]
        #                       for i in range(length)]
        # Convert true/false to +1/-1.
        # pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.int32)
        pm_meas = 1 - 2 * np.array(measurements).astype(np.int32)

        tot_energy = np.sum(pm_meas * np.reshape(h, length*length))
        for i, jr_row in enumerate(jr):
            for j, jr_ij in enumerate(jr_row):
                tot_energy += jr_ij * pm_meas[i*length+j] * pm_meas[(i+1)*length+j]
        for i, jc_row in enumerate(jc):
            for j, jc_ij in enumerate(jc_row):
                tot_energy += jc_ij * pm_meas[i*length+j] * pm_meas[i*length+j+1]
        return tot_energy
    return energy

def obj_func(result, h, jr, jc):
    length = len(h)
    energy_hist = result.histogram(key='x', fold_func=energy_func(length, h, jr, jc))
    return np.sum([k * v for k,v in energy_hist.items()]) / result.repetitions

def rot_x_layer(length, half_turns):
    """Yields X rotations by half_turns on a square grid of given length."""
    rot = cirq.XPowGate(exponent=half_turns)
    for i in range(length):
        for j in range(length):
            yield rot(cirq.LineQubit(i*length+j))

def rot_z_layer(h, half_turns):
    """Yields Z rotations by half_turns conditioned on the field h."""
    length = len(h)
    gate = cirq.ZPowGate(exponent=half_turns)
    for i, h_row in enumerate(h):
        for j, h_ij in enumerate(h_row):
            if h_ij == 1:
                yield gate(cirq.LineQubit(i*length+j))

def rot_11_layer(length, jr, jc, half_turns):
    """Yields rotations about |11> conditioned on the jr and jc fields."""
    gate = cirq.CZPowGate(exponent=half_turns)
    for i, jr_row in enumerate(jr):
        for j, jr_ij in enumerate(jr_row):
            if jr_ij == -1:
                yield cirq.X(cirq.LineQubit(i*length+j))
                yield cirq.X(cirq.LineQubit((i+1)*length+j))
            yield gate(cirq.LineQubit(i*length+j),
                       cirq.LineQubit((i+1)*length+j))
            if jr_ij == -1:
                yield cirq.X(cirq.LineQubit(i*length+j))
                yield cirq.X(cirq.LineQubit((i+1)*length+j))

    for i, jc_row in enumerate(jc):
        for j, jc_ij in enumerate(jc_row):
            if jc_ij == -1:
                yield cirq.X(cirq.LineQubit(i*length+j))
                yield cirq.X(cirq.LineQubit(i*length+j+1))
            yield gate(cirq.LineQubit(i*length+j),
                       cirq.LineQubit(i*length+j+1))
            if jc_ij == -1:
                yield cirq.X(cirq.LineQubit(i*length+j))
                yield cirq.X(cirq.LineQubit(i*length+j+1))

def one_step(h, jr, jc, x_half_turns, h_half_turns, j_half_turns):
    length = len(h)
    yield rot_x_layer(length, x_half_turns)
    yield rot_z_layer(h, h_half_turns)
    yield rot_11_layer(length, jr, jc, j_half_turns)

if __name__ == '__main__':
    main()
