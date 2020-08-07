import random
import math
import numpy as np
import sympy
import scipy.optimize

import cirq
import qsimcirq
import time

from collections import defaultdict

def main(
    repetitions=256,
    maxiter=1
    ):
    # Set problem parameters
    kc_kl_divs={}
    qs_kl_divs={}
    for length in range (2,6,1):

        kc_kl_divs[length]=[]
        qs_kl_divs[length]=[]

        for print_length in range(2,length,1):
            for kc_kl_div, qs_kl_div in zip(kc_kl_divs[print_length],qs_kl_divs[print_length]):
                print("print_length="+str(print_length)+" kc_kl_div="+str(kc_kl_div)+" qs_kl_div="+str(qs_kl_div))

        for _ in range(16):

            steps = 1

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
            qs_sim_16 = qsimcirq.QSimSimulator( qsim_options={'t':16, 'v': 0} )
            kc_sim = cirq.KnowledgeCompilationSimulator(cirq_circuit, initial_state=0)
            kc_smp = cirq.KnowledgeCompilationSimulator(meas_circuit, initial_state=0)

            iter = 0
            def f(x):
                param_resolver = cirq.ParamResolver({ 'alpha':x[0], 'beta':x[1], 'gamma':x[2] })

                # VALIDATE STATE VECTOR SIMULATION
                solved_circuit = cirq.resolve_parameters(meas_circuit, param_resolver)
                cirq.ConvertToCzAndSingleGates().optimize_circuit(solved_circuit) # cannot work with params
                cirq.ExpandComposite().optimize_circuit(solved_circuit)
                qsim_circuit = qsimcirq.QSimCircuit(solved_circuit)

                qs_sim_start = time.time()
                qs_sim_result = qs_sim_16.simulate(qsim_circuit)
                qs_sim_time = time.time() - qs_sim_start

                # kc_sim_start = time.time()
                # kc_sim_result = kc_sim.simulate(cirq_circuit, param_resolver=param_resolver)
                # kc_sim_time = time.time() - kc_sim_start

                # print("kc_sim_result.state_vector()=")
                # print(kc_sim_result.state_vector())
                # print("qs_sim_result.state_vector()=")
                # print(qs_sim_result.state_vector())

                # assert qs_sim_result.state_vector().shape == (1<<(length*length),)
                # assert cirq.linalg.allclose_up_to_global_phase(
                #     qs_sim_result.state_vector(),
                #     kc_sim_result.state_vector(),
                #     rtol = 1.e-4,
                #     atol = 1.e-6,
                # )

                # VALIDATE SAMPLING HISTOGRAMS

                # Sample bitstrings from circuit
                qs_smp_16_start = time.time()
                qs_smp_16_result = qs_sim_16.run(qsim_circuit, repetitions=repetitions)
                qs_smp_16_time = time.time() - qs_smp_16_start

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

                # Sample bitstrings from circuit
                kc_smp_start = time.time()
                kc_smp_result = kc_smp.run(meas_circuit, param_resolver=param_resolver, repetitions=repetitions)
                kc_smp_time = time.time() - kc_smp_start

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

                nonlocal iter
                # PRINT HISTOGRAMS
                kc_kl_div = 0
                qs_kl_div = 0
                print ('iter,index,bitstring,bitstring_bin,qs_probability,qs_samples,kc_samples')
                probabilities = np.zeros(1<<(length*length))
                for bitstring, amplitude in enumerate(qs_sim_result.state_vector()):
                    probability = abs(amplitude) * abs(amplitude)
                    kc_samples = kc_histogram[bitstring]/repetitions
                    qs_samples = qs_histogram[bitstring]/repetitions
                    kc_kl_div += 0 if kc_samples==0 else kc_samples*math.log(kc_samples/probability)
                    qs_kl_div += 0 if qs_samples==0 else qs_samples*math.log(qs_samples/probability)
                    probabilities[bitstring]=probability
                kc_kl_divs[length].append(kc_kl_div)
                qs_kl_divs[length].append(qs_kl_div)
                sorted_bitstrings = np.argsort(probabilities)
                # for index, bitstring in enumerate(sorted_bitstrings):
                #     print (str(iter)+','+str(index)+','+str(bitstring)+','+format(bitstring,'b').zfill(length*length)+','+str(probabilities[bitstring])+','+"{:.6e}".format(qs_histogram[bitstring]/repetitions)+','+"{:.6e}".format(kc_histogram[bitstring]/repetitions))

                print ('qs_value='+str(qs_value)+' kc_value='+str(kc_value))
                # print ( 'qs_sim_time='+str(qs_sim_time)+' kc_sim_time='+str(kc_sim_time) )
                print ( 'qs_smp_16_time='+str(qs_smp_16_time)+' kc_smp_time='+str(kc_smp_time) )
                print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                iter += 1
                return qs_value

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
