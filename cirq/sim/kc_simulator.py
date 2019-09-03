# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simulator that uses Bayesian network knowledge compilation."""

import collections

from typing import Any, Dict, Iterator, List, Union

import numbers, cmath, sympy
import numpy as np

from cirq import circuits, ops, protocols, schedules, study, optimizers
from cirq.sim import simulator, wave_function, wave_function_simulator, sparse_simulator

import os, subprocess, re, csv, sys, time

class KnowledgeCompilationSimulator(simulator.SimulatesSamples,
                                    wave_function_simulator.SimulatesIntermediateWaveFunction):
    """A wave function simulator based on Bayesian network knowledge compilation.

    This simulator can be applied on circuits that are made up of operations
    that have a `_unitary_` method, or `_has_unitary_` and
    `_apply_unitary_`, `_mixture_` methods, are measurements, or support a
    `_decompose_` method that returns operations satisfying these same
    conditions. That is to say, the operations should follow the
    `cirq.SupportsApplyUnitary` protocol, the `cirq.SupportsUnitary` protocol,
    the `cirq.SupportsMixture` protocol, or the `cirq.CompositeOperation`
    protocol. It is also permitted for the circuit to contain measurements
    which are operations that support `cirq.SupportsChannel` and
    `cirq.SupportsMeasurementKey`

    This simulator supports three types of simulation.

    Run simulations which mimic running on actual quantum hardware. These
    simulations do not give access to the wave function (like actual hardware).
    There are two variations of run methods, one which takes in a single
    (optional) way to resolve parameterized circuits, and a second which
    takes in a list or sweep of parameter resolver:

        run(circuit, param_resolver, repetitions)

        run_sweep(circuit, params, repetitions)

    The simulation performs optimizations if the number of repetitions is
    greater than one and all measurements in the circuit are terminal (at the
    end of the circuit). These methods return `TrialResult`s which contain both
    the measurement results, but also the parameters used for the parameterized
    circuit operations. The initial state of a run is always the all 0s state
    in the computational basis.

    By contrast the simulate methods of the simulator give access to the
    wave function of the simulation at the end of the simulation of the circuit.
    These methods take in two parameters that the run methods do not: a
    qubit order and an initial state. The qubit order is necessary because an
    ordering must be chosen for the kronecker product (see
    `SparseSimulationTrialResult` for details of this ordering). The initial
    state can be either the full wave function, or an integer which represents
    the initial state of being in a computational basis state for the binary
    representation of that integer. Similar to run methods, there are two
    simulate methods that run for single runs or for sweeps across different
    parameters:

        simulate(circuit, param_resolver, qubit_order, initial_state)

        simulate_sweep(circuit, params, qubit_order, initial_state)

    The simulate methods in contrast to the run methods do not perform
    repetitions. The result of these simulations is a
    `SparseSimulationTrialResult` which contains, in addition to measurement
    results and information about the parameters that were used in the
    simulation,access to the state via the `state` method and `StateVectorMixin`
    methods.

    If one wishes to perform simulations that have access to the
    wave function as one steps through running the circuit there is a generator
    which can be iterated over and each step is an object that gives access
    to the wave function.  This stepping through a `Circuit` is done on a
    `Moment` by `Moment` manner.

        simulate_moment_steps(circuit, param_resolver, qubit_order,
                              initial_state)

    One can iterate over the moments via

        for step_result in simulate_moments(circuit):
           # do something with the wave function via step_result.state

    Note also that simulations can be stochastic, i.e. return different results
    for different runs.  The first version of this occurs for measurements,
    where the results of the measurement are recorded.  This can also
    occur when the circuit has mixtures of unitaries.

    Finally, one can compute the values of displays (instances of
    `SamplesDisplay` or `WaveFunctionDisplay`) in the circuit:

        compute_displays(circuit, param_resolver, qubit_order, initial_state)

        compute_displays_sweep(circuit, params, qubit_order, initial_state)

    The result of computing display values is stored in a
    `ComputeDisplaysResult`.

    See `Simulator` for the definitions of the supported methods.
    """

    def _unitary_to_transposed_cpt ( self, gate, unitary ):
        assert all (len (row) == len (unitary) for row in unitary)
        if len(unitary)==2:
            stencil = []
            stencil.append([[1,0],[0,1]])
            return np.dot(stencil,unitary)
        # For two-qubit gates and beyond, check that unitaries are permutation matrices or it's wrong
        assert all (count==1 for count in np.count_nonzero(unitary, axis=0))
        assert all (count==1 for count in np.count_nonzero(unitary, axis=1))
        # CNot and CZ gates only need one CPT on target qubit to represent the gates
        # Furthermore, CZ gates can swap the control and target qubits (not yet implemented)
        if len(unitary)==4 and isinstance( gate, ( ops.CNotPowGate, ops.CZPowGate ) ):
            stencil = []
            stencil.append([[1,0,1,0],[0,1,0,1]])
            return np.dot ( stencil, unitary )
        elif len(unitary)==4:
            stencil = []
            stencil.append([[1,1,0,0],[0,0,1,1]])
            stencil.append([[1,0,1,0],[0,1,0,1]])
            # return np.dot(stencil,np.sqrt(unitary)) # distributes phase across both cpts
            # Need to be able to take square root of both numerical values and symbols
            return np.dot ( stencil,
                [ [cmath.sqrt(element) if isinstance ( element, numbers.Number ) else sympy.sqrt(element)
                    for element in row] for row in unitary] ) # distributes phase across both cpts
        raise Exception(f'Size {len(unitary)} unitary unsupported.')

    net_prelude = \
'''node {target_posterior} {{
    states = ("|0>" "|1>");
    subtype = boolean;
}}
potential ( {target_posterior} | '''

    net_interlude = \
''') {{
    data = '''

    def _net_data_format ( self, parents, assignments ):
        out_string = '('
        if parents:
            out_string += self._net_data_format ( parents[1:], 2*assignments+0 )
            out_string += self._net_data_format ( parents[1:], 2*assignments+1 )
        else:
            out_string += '{data['+str(assignments)+'][0]} {data['+str(assignments)+'][1]}'
        out_string += ')'
        return out_string

    # If CPT element is complex number, writes CPP style complex value for bayes-to-cnf
    # If CPT element is a sympy object, writes a 28-bit integer hash so we can find it again at inference time
    def _to_cpp_complex_hash ( self, complex_symbols ):
        if complex_symbols==0:
            return '0'
        elif isinstance ( complex_symbols, numbers.Number ) :
            return(f'{complex_symbols.real:.8f},{complex_symbols.imag:.8f}')
        else:
            self._hash_to_symbols[hash(complex_symbols)%(1<<28)] = complex_symbols
            return(hash(complex_symbols)%(1<<28))

    def _cpt_to_cpp_complex_hash ( self, cpt ):
        if hasattr(cpt,'__len__'):
            return [ self._cpt_to_cpp_complex_hash(element) for element in cpt ]
        else:
            return self._to_cpp_complex_hash ( cpt )

    net_postlude = \
''';
}}

'''

    def _to_java_complex ( self, complex_symbols ):
        if complex_symbols==0:
            return '0'
        return(f'{complex_symbols.real:f}+{complex_symbols.imag:f}i')

    def __init__(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
        intermediate: bool = False, # whether or not moment steps, intermediate measurement allowed.
        dtype=np.complex64):
        """A sparse matrix simulator.

        Args:
            dtype: The `numpy.dtype` used by the simulation. One of
            `numpy.complex64` or `numpy.complex128`
        """
        if np.dtype(dtype).kind != 'c':
            raise ValueError(
                'dtype must be a complex type but was {}'.format(dtype))
        self._dtype = dtype
        self._intermediate = intermediate

        circuit = (program if isinstance(program, circuits.Circuit) else program.to_circuit())

        if not self._intermediate: # messes up moment steps, moment step samping
            optimizers.ExpandComposite().optimize_circuit(circuit)
            # optimizers.ConvertToCzAndSingleGates().optimize_circuit(circuit) # cannot work with params
            optimizers.MergeInteractions().optimize_circuit(circuit)
            optimizers.MergeSingleQubitGates().optimize_circuit(circuit)
            optimizers.DropEmptyMoments().optimize_circuit(circuit)
            optimizers.EjectPhasedPaulis().optimize_circuit(circuit)
            pass
        optimizers.EjectZ().optimize_circuit(circuit)
        # optimizers.DropNegligible().optimize_circuit(circuit)

        self._circuit = circuit
        self._qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        self._num_qubits = len(self._qubits)
        self._qubit_map = {q: i for i, q in enumerate(self._qubits)}

        net_file = open('circuit.net', 'w')
        net_file.write('net {}\n\n')

        # initial_state: Union[int, np.ndarray],
        self._initial_state_lockout = False if initial_state is None else True
        # generate Bayesian network nodes with no priors for qubit initialization
        qubit_to_last_moment_index = {}
        actual_initial_state = 0 if initial_state is None else initial_state
        for target_qubit, initial_value in zip (
            self._qubits,
            # to adhere to Cirq's endian convention:
            [bool(actual_initial_state & (1<<n)) for n in reversed(range(self._num_qubits))]
            ):

            qubit_to_last_moment_index[target_qubit] = 0

            target_posterior =  'n' + str(qubit_to_last_moment_index[target_qubit]).zfill(4) + 'q' + str(target_qubit).zfill(4)
            node_string = self.net_prelude
            node_string += self.net_interlude

            if initial_state is not None:
                node_string += '({} {})'.format (
                    '0' if initial_value else '1',
                    '1' if initial_value else '0'
                    )
            else:
                qi0_sym = 'i0' + 'q'+str(target_qubit).zfill(4)
                qi1_sym = 'i1' + 'q'+str(target_qubit).zfill(4)
                node_string += '({} {})'.format ( hash(qi0_sym)%(1<<28), hash(qi1_sym)%(1<<28) )

            node_string += self.net_postlude

            net_file.write(node_string.format(target_posterior=target_posterior))

        self._hash_to_symbols = {}
        for moment_index, moment in enumerate(circuit, start=1):
            for op in moment:

                if isinstance(op,(ops.PauliString, ops.PauliStringPhasor)) or \
                not isinstance(op,(ops.PauliStringExpectation, ops.ApproxPauliStringExpectation)) and not isinstance(op.gate,ops.MeasurementGate):

                    if isinstance(op,(ops.PauliString,ops.PauliStringPhasor)):
                        gate = None
                    else:
                        gate = op.gate

                    if protocols.has_mixture(op):
                        unitary_matrix = [[0+0j,0+0j],[0+0j,0+0j]]
                        for component in protocols.mixture(op):
                            unitary_matrix += component[0] * component[1]
                    else:
                        unitary_matrix = protocols.unitary(op)

                    transposed_cpts = self._unitary_to_transposed_cpt( gate, unitary_matrix )

                    for target_qubit, transposed_cpt in zip(reversed(op.qubits), reversed(transposed_cpts)):

                        node_string = self.net_prelude

                        parents=[]
                        for control_qubit in op.qubits:
                            depth = str(qubit_to_last_moment_index[control_qubit]).zfill(4)
                            parent = 'n' + depth + 'q' + str(control_qubit).zfill(4)
                            node_string += parent + ' '
                            parents.append(parent)
                        node_string += self.net_interlude
                        node_string += self._net_data_format ( parents, 0 )
                        node_string += self.net_postlude

                        target_posterior = 'n' + str(moment_index).zfill(4) + 'q' + str(target_qubit).zfill(4)
                        net_file.write(node_string.format(
                            target_posterior=target_posterior,
                            data=self._cpt_to_cpp_complex_hash(transposed_cpt.transpose())
                            ))

                    # update depth
                    for target_qubit, transposed_cpt in zip(reversed(op.qubits), reversed(transposed_cpts)):
                        qubit_to_last_moment_index[target_qubit] = moment_index

        net_file.close()

        # Bayesian network to conjunctive normal form
        # TODO: autoinstall this
        stdout = os.system('/n/fs/qdb/bayes-to-cnf/bin/bn-to-cnf -d -a -b -i circuit.net -w -s')
        # -e: Equal probabilities are encoded is incompatible with dtbnorders
        # -e and -b used together causes moment steps simulation to fail
        print (stdout)

        self._node_re_compile = re.compile(r'cc\$I\$(\d+)\$1.0\$\+\$n(\d+)q(\d+)') # are negative literals and opt bool valid?
        self._int_re_compile = re.compile(r'cc\$C\$\d+\$(\d+)')
        existentially_quantified_variables = []
        with open('circuit.cnf', 'r') as cnf_file:
            with open('circuit.lmap', 'w') as lmap_file:
                for line in cnf_file:
                    if line.startswith('cc'):
                        lmap_file.write(line)
                        match = self._node_re_compile.match(line)
                        if match:
                            moment = int(match.group(2))
                            qubit = self._qubits[int(match.group(3))]
                            if moment<qubit_to_last_moment_index[qubit]:
                                existentially_quantified_variables.append(match.group(1))

        if not self._intermediate:
            with open('variables.file', 'w') as exist_file:
                line = str(len(existentially_quantified_variables))
                for variable in existentially_quantified_variables:
                    line += ' ' + variable
                exist_file.write(line)

        try:
            # Conjunctive normal form to arithmetic circuit
            bestFileSize = sys.maxsize
            for _ in range(1):
                stdout = os.system('/n/fs/qdb/qACE/ace_v3.0_linux86/c2d_linux -simplify_s -in circuit.cnf')
                if not self._intermediate:
                    stdout = os.system('/n/fs/qdb/qACE/ace_v3.0_linux86/c2d_linux -exist variables.file -reduce -in circuit.cnf_simplified')
                else:
                    stdout = os.system('/n/fs/qdb/qACE/ace_v3.0_linux86/c2d_linux -dt_method 3 -reduce -in circuit.cnf_simplified')
                # stdout = os.system('/n/fs/qdb/qACE/miniC2D-1.0.0/bin/linux/miniC2D -c circuit.cnf_simplified')
                print (stdout)
                currFileSize = os.path.getsize('circuit.cnf_simplified.nnf')
                if currFileSize<bestFileSize:
                    bestFileSize = currFileSize
                    os.rename('circuit.cnf_simplified.nnf','best.cnf_simplified.nnf')
            os.rename('best.cnf_simplified.nnf','circuit.cnf_simplified.nnf')

            # Build the evaluator for the arithmetic circuit
            stdout = os.system('mkdir evaluator')
            stdout = os.system('javac -d evaluator -cp /n/fs/qdb/qACE/commons-math3-3.6.1/commons-math3-3.6.1.jar -Xlint:unchecked /n/fs/qdb/Cirq/cirq/sim/Evaluator.java /n/fs/qdb/qACE/org/apache/commons/math3/complex/ComplexFormat.java /n/fs/qdb/qACE/aceEvalComplexSrc/OnlineEngine.java /n/fs/qdb/qACE/aceEvalComplexSrc/Calculator.java /n/fs/qdb/qACE/aceEvalComplexSrc/Evidence.java /n/fs/qdb/qACE/aceEvalComplexSrc/OnlineEngineSop.java /n/fs/qdb/qACE/aceEvalComplexSrc/CalculatorNormal.java /n/fs/qdb/qACE/aceEvalComplexSrc/CalculatorLogE.java /n/fs/qdb/qACE/aceEvalComplexSrc/UnderflowException.java')
            print (stdout)

            # Launch the evaluator in a subprocess
            self._subprocess = subprocess.Popen(["java", "-cp", "evaluator:/n/fs/qdb/qACE/commons-math3-3.6.1/commons-math3-3.6.1.jar", "edu.ucla.belief.ace.Evaluator", "circuit.lmap", "circuit.cnf_simplified.nnf", str(self._num_qubits)], stdin=subprocess.PIPE)

        except:
            pass

    def __del__(self):
        self._subprocess.kill()

    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        """See definition in `cirq.SimulatesSamples`."""
        param_resolver = param_resolver or study.ParamResolver({})
        def measure_or_mixture(op):
            return protocols.is_measurement(op) or protocols.has_mixture(op)
        if circuit.are_all_matches_terminal(measure_or_mixture):
            return self._run_sweep_sample(circuit, param_resolver, repetitions)
        else:
            if not self._intermediate:
                raise Exception(f'KnowledgeCompilationSimulator not properly configured for intermediate state simulation.')
            return self._run_sweep_repeat(param_resolver, repetitions)

    def _run_sweep_sample(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, List[np.ndarray]]:

        self._repetitions = repetitions
        self._subprocess.stdin.write(f'cc$R${self._repetitions}\n'.encode())

        measurements = {}  # type: Dict[str, List[np.ndarray]]

        all_step_results = self._base_iterator(
                param_resolver=param_resolver,
                initial_state=0)

        for step_result in all_step_results:
            pass
        return step_result.measurements

    def _run_sweep_repeat(
        self,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, List[np.ndarray]]:

        if not self._intermediate:
            raise Exception(f'KnowledgeCompilationSimulator not properly configured for intermediate state simulation.')

        measurements = {}  # type: Dict[str, List[np.ndarray]]
        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                    param_resolver=param_resolver,
                    initial_state=0)

            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if not k in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=bool))
        return {k: np.array(v) for k, v in measurements.items()}

    def compute_amplitudes_sweep(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            bitstrings: np.ndarray,
            params: study.Sweepable,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ) -> List[List[complex]]:

        self._amplitude_indices = [
            wave_function_simulator._bitstring_to_integer(bitstring) for bitstring in bitstrings
        ]

        for amplitude_index in self._amplitude_indices:
            self._subprocess.stdin.write(f'cc$A${amplitude_index}\n'.encode())

        return wave_function_simulator.SimulatesIntermediateWaveFunction.compute_amplitudes_sweep(
            self, program, bitstrings, params, qubit_order)

    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> List['SimulationTrialResult']:
        """Simulates the supplied Circuit or Schedule.

        This method returns a result which allows access to the entire
        wave function. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            List of SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """
        self._simulate_sweep_flag = None
        return simulator.SimulatesIntermediateState.simulate_sweep(
            self,
            program,
            params,
            qubit_order,
            initial_state)

    def _simulator_iterator(
            self,
            circuit: circuits.Circuit, # unused
            param_resolver: study.ParamResolver,
            qubit_order: ops.QubitOrderOrList, # unused
            initial_state: Union[int, np.ndarray],
    ) -> Iterator:
        """See definition in `cirq.SimulatesIntermediateState`.

        If the initial state is an int, the state is set to the computational
        basis state corresponding to this state. Otherwise  if the initial
        state is a np.ndarray it is the full initial state. In this case it
        must be the correct size, be normalized (an L2 norm of 1), and
        be safely castable to an appropriate dtype for the simulator.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        if self._initial_state_lockout and initial_state is not None:
            raise Exception(f'Do not supply initial_state in both initialization and simulation.')
        actual_initial_state = 0 if initial_state is None else initial_state
        return self._base_iterator(param_resolver,
                                   actual_initial_state,
                                   perform_measurements=True)

    def _base_iterator(
            self,
            param_resolver: study.ParamResolver,
            initial_state: Union[int, np.ndarray],
            perform_measurements: bool=True,
    ) -> Iterator:

        if len(self._circuit) == 0:
            yield sparse_simulator.SparseSimulatorStep(
                wave_function.to_valid_state_vector(initial_state,self._num_qubits,self._dtype),
                {},
                self._qubit_map,
                self._dtype)

        else:

            prep_start = time.time()

            param_dict = {}
            for target_qubit, initial_value in zip (
                self._qubits,
                # to adhere to Cirq's endian convention:
                [bool(initial_state & (1<<n)) for n in reversed(range(self._num_qubits))]
                ):
                qi0_sym = 'i0'+'q'+str(target_qubit).zfill(4)
                param_dict[ hash(qi0_sym)%(1<<28) ] = '0' if initial_value else '1'
                qi1_sym = 'i1'+'q'+str(target_qubit).zfill(4)
                param_dict[ hash(qi1_sym)%(1<<28) ] = '1' if initial_value else '0'

            param_resolver = param_resolver or study.ParamResolver({})
            hash_csv = int(hash((self._subprocess, param_resolver)))
            for hash_key, symbols in self._hash_to_symbols.items():
                param_dict[hash_key] = self._to_java_complex(protocols.resolve_parameters(symbols, param_resolver))

            print("prep time = ")
            print(time.time() - prep_start)

            if hasattr(self,'_simulate_sweep_flag') or hasattr(self,'_repetitions'):
                for last_moment_index, moment in enumerate(self._circuit, start=1):
                    pass
            else:
                if not self._intermediate:
                    raise Exception(f'KnowledgeCompilationSimulator not properly configured for intermediate state simulation.')

            for moment_index, moment in enumerate(self._circuit, start=1):

                if (hasattr(self,'_simulate_sweep_flag') or hasattr(self,'_repetitions')) and moment_index != last_moment_index:
                    pass
                else:

                    lmap_start = time.time()

                    csv_basename = f'{hash_csv}_{initial_state:04d}_{moment_index:04d}'
                    self._subprocess.stdin.write(f'cc$B${csv_basename}\n'.encode())
                    self._subprocess.stdin.write(f'cc$M${moment_index}\n'.encode())

                    with open('circuit.lmap', 'r') as lmap_file:
                        for line in lmap_file:

                            match = self._int_re_compile.match(line)
                            if match and int(match.group(1)) in param_dict:
                                line = re.sub(match.group(1), param_dict[int(match.group(1))], line)

                            if self._intermediate:
                                match = self._node_re_compile.match(line)
                                if match and int(match.group(2))>moment_index:
                                    line = re.sub(r'\+', 'I', line)

                            self._subprocess.stdin.write(line.encode())

                    measurements = collections.defaultdict(
                            list)  # type: Dict[str, List[bool]]

                    print("lmap time = ")
                    print(time.time() - lmap_start)
                    java_start = time.time()

                    csv_name = f'{csv_basename}.csv'
                    while not os.path.exists(csv_name):
                        self._subprocess.stdin.write(b'\n') # keep pushing the BufferedReader

                    print("java time = ")
                    print(time.time() - java_start)
                    post_start = time.time()

                    state_vector = []
                    outputQubitString = 0
                    with open(csv_name, 'r') as csv_file:
                        if hasattr(self, '_repetitions'):
                            for outputQubitString in range(1<<self._num_qubits):
                                state_vector.append(0)
                            def tobin(x,s):
                                return [(x>>k)&1 for k in reversed(range(0,s))]
                            for repetition in range(self._repetitions):
                                line = csv_file.readline()
                                row = line.split(',')
                                # print (row)
                                # print (row[0])
                                bin_list = tobin(int(row[0]),self._num_qubits)
                                for op in moment:
                                    indices = [self._qubit_map[qubit] for qubit in op.qubits]
                                    if protocols.is_measurement(op):
                                        meas = ops.op_gate_of_type(op, ops.MeasurementGate)
                                        if meas:
                                            invert_mask = meas.full_invert_mask()
                                            bits = [bin_list[index] for index in indices]
                                            corrected = [bit ^ mask for bit, mask in zip(bits, invert_mask)]
                                            key = protocols.measurement_key(meas)
                                            measurements[key].append(corrected)
                        else:
                            for outputQubitString in range(1<<self._num_qubits):
                                if hasattr(self, '_amplitude_indices'):
                                    if outputQubitString in self._amplitude_indices:
                                        line = csv_file.readline()
                                        row = line.split(',')
                                        # print (row)
                                        # print (row[0])
                                        assert int(row[0]) == outputQubitString
                                        state_vector.append(complex(row[1]))
                                    else:
                                        state_vector.append(0)
                                else:
                                    line = csv_file.readline()
                                    row = line.split(',')
                                    # print (row)
                                    # print (row[0])
                                    assert int(row[0]) == outputQubitString
                                    state_vector.append(complex(row[1]))
                    # assert float(row[2])-1.0 < 1.0/256.0
                    os.remove(csv_name)

                    if not hasattr(self, '_repetitions'):
                        for op in moment:
                            indices = [self._qubit_map[qubit] for qubit in op.qubits]
                            if protocols.is_measurement(op):
                                # Do measurements second, since there may be mixtures that
                                # operate as measurements.
                                # TODO: support measurement outside the computational basis.
                                if perform_measurements:
                                    self._simulate_measurement(op, np.reshape(state_vector, (2,) * self._num_qubits), indices,
                                                               measurements, self._num_qubits)

                    print("post time = ")
                    print(time.time() - post_start)

                    yield sparse_simulator.SparseSimulatorStep(
                        state_vector=state_vector,
                        measurements=measurements,
                        qubit_map=self._qubit_map,
                        dtype=self._dtype)

    def _simulate_measurement(self, op: ops.Operation, data,
            indices: List[int], measurements: Dict[str, List[bool]],
            num_qubits: int) -> None:
        """Simulate an op that is a measurement in the computataional basis."""
        meas = ops.op_gate_of_type(op, ops.MeasurementGate)
        # TODO: support measurement outside computational basis.
        if meas:
            invert_mask = meas.full_invert_mask()
            # Measure updates inline.
            bits, _ = wave_function.measure_state_vector(data,
                                                         indices,
                                                         data)
            corrected = [bit ^ mask for bit, mask in zip(bits, invert_mask)]
            key = protocols.measurement_key(meas)
            measurements[key].extend(corrected)
