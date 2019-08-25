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

from typing import Any, List, Union

import numbers, cmath, sympy
import numpy as np

from cirq import circuits, ops, protocols, schedules, study, optimizers
from cirq.sim import simulator, wave_function_simulator

import os, subprocess, re, csv, sys, time

class KnowledgeCompilationSimulator(simulator.SimulatesFinalState):
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
    _hash_to_symbols = {}
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
        elif isinstance ( complex_symbols, numbers.Number ) :
            return(f'{complex_symbols.real:f}+{complex_symbols.imag:f}i')
        else:
            raise Exception(f'Not a number: {complex_symbols}.')

    def __init__(
        self,
        program: Union[circuits.Circuit, schedules.Schedule],
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Any = None,
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

        circuit = (program if isinstance(program, circuits.Circuit) else program.to_circuit())

        optimizers.ExpandComposite().optimize_circuit(circuit)
        # optimizers.ConvertToCzAndSingleGates().optimize_circuit(circuit) # cannot work with params
        # optimizers.EjectPhasedPaulis().optimize_circuit(circuit)
        optimizers.EjectZ().optimize_circuit(circuit)
        # optimizers.MergeInteractions().optimize_circuit(circuit)
        optimizers.MergeSingleQubitGates().optimize_circuit(circuit)
        optimizers.DropEmptyMoments().optimize_circuit(circuit)
        # optimizers.DropNegligible().optimize_circuit(circuit)

        self._qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
        self._num_qubits = len(self._qubits)
        self._qubit_map = {q: i for i, q in enumerate(self._qubits)}

        net_file = open('circuit.net', 'w')
        net_file.write('net {}\n\n')

        # initial_state: Union[int, np.ndarray],
        self._initial_state_lockout = False if initial_state is None else True
        # generate Bayesian network nodes with no priors for qubit initialization
        # TODO: this might need to be indexed by moments
        qubit_to_last_gate_index = {}
        actual_initial_state = 0 if initial_state is None else initial_state
        for target_qubit, initial_value in zip (
            self._qubits,
            # to adhere to Cirq's endian convention:
            [bool(actual_initial_state & (1<<n)) for n in reversed(range(self._num_qubits))]
            ):

            qubit_to_last_gate_index[target_qubit] = 0

            target_posterior = 'q' + str(target_qubit).zfill(4) + 'n' + str(qubit_to_last_gate_index[target_qubit]).zfill(4)
            node_string = self.net_prelude
            node_string += self.net_interlude

            if initial_state is not None:
                node_string += '({} {})'.format (
                    '0' if initial_value else '1',
                    '1' if initial_value else '0'
                    )
            else:
                qi0_sym = 'q'+str(target_qubit).zfill(4)+'i0'
                qi1_sym = 'q'+str(target_qubit).zfill(4)+'i1'
                node_string += '({} {})'.format ( hash(qi0_sym)%(1<<28), hash(qi1_sym)%(1<<28) )

            node_string += self.net_postlude

            net_file.write(node_string.format(target_posterior=target_posterior))

        dag = circuits.CircuitDag.from_circuit(circuit)
        for posterior_node in dag.ordered_nodes():

            transposed_cpts = self._unitary_to_transposed_cpt( posterior_node.val.gate, protocols.unitary(posterior_node.val) )

            for target_qubit, transposed_cpt in zip(reversed(posterior_node.val.qubits), reversed(transposed_cpts)):

                node_string = self.net_prelude

                parents=[]
                for control_qubit in posterior_node.val.qubits:
                    depth = str(qubit_to_last_gate_index[control_qubit]).zfill(4)
                    parent = 'q' + str(control_qubit).zfill(4) + 'n' + depth
                    node_string += parent + ' '
                    parents.append(parent)
                node_string += self.net_interlude
                node_string += self._net_data_format ( parents, 0 )
                node_string += self.net_postlude

                target_posterior = 'q' + str(target_qubit).zfill(4) + 'n' + str(qubit_to_last_gate_index[target_qubit]+1).zfill(4)
                net_file.write(node_string.format(
                    target_posterior=target_posterior,
                    data=self._cpt_to_cpp_complex_hash(transposed_cpt.transpose())
                    ))

            # update depth
            for target_qubit, transposed_cpt in zip(reversed(posterior_node.val.qubits), reversed(transposed_cpts)):
                qubit_to_last_gate_index[target_qubit] += 1

        net_file.close()

        # Bayesian network to conjunctive normal form
        # TODO: autoinstall this
        stdout = os.system('/n/fs/qdb/bayes-to-cnf/bin/bn-to-cnf -e -d -a -i circuit.net -w -s')
        print (stdout)

        with open('circuit.cnf', 'r') as cnf_file:
            with open('circuit.lmap', 'w') as lmap_file:
                for line in cnf_file:
                    if line.startswith('cc'):
                        lmap_file.write(line)
        self._re_compile = re.compile(r'[-+]?\d+')

        # Conjunctive normal form to arithmetic circuit
        bestFileSize = sys.maxsize
        for _ in range(4):
            stdout = os.system('/n/fs/qdb/qACE/ace_v3.0_linux86/c2d_linux -reduce -in circuit.cnf')
            # stdout = os.system('/n/fs/qdb/qACE/ace_v3.0_linux86/c2d_linux -reduce -minimize -smooth_all -keep_trivial_cls -in circuit.cnf')
            # stdout = os.system('/n/fs/qdb/qACE/miniC2D-1.0.0/bin/linux/miniC2D -c circuit.cnf')
            print (stdout)
            currFileSize = os.path.getsize('circuit.cnf.nnf')
            if currFileSize<bestFileSize:
                bestFileSize = currFileSize
                os.rename('circuit.cnf.nnf','best.cnf.nnf')
        os.rename('best.cnf.nnf','circuit.cnf.nnf')

        # Build the evaluator for the arithmetic circuit
        stdout = os.system('mkdir evaluator')
        stdout = os.system('javac -d evaluator -cp /n/fs/qdb/qACE/commons-math3-3.6.1/commons-math3-3.6.1.jar -Xlint:unchecked /n/fs/qdb/Cirq/cirq/sim/Evaluator.java /n/fs/qdb/qACE/org/apache/commons/math3/complex/ComplexFormat.java /n/fs/qdb/qACE/aceEvalComplexSrc/OnlineEngine.java /n/fs/qdb/qACE/aceEvalComplexSrc/Calculator.java /n/fs/qdb/qACE/aceEvalComplexSrc/Evidence.java /n/fs/qdb/qACE/aceEvalComplexSrc/OnlineEngineSop.java /n/fs/qdb/qACE/aceEvalComplexSrc/CalculatorNormal.java /n/fs/qdb/qACE/aceEvalComplexSrc/CalculatorLogE.java /n/fs/qdb/qACE/aceEvalComplexSrc/UnderflowException.java')
        print (stdout)

        # Launch the evaluator in a subprocess
        self._subprocess = subprocess.Popen(["java", "-cp", "evaluator:/n/fs/qdb/qACE/commons-math3-3.6.1/commons-math3-3.6.1.jar", "edu.ucla.belief.ace.Evaluator", "circuit.lmap", "circuit.cnf.nnf", str(self._num_qubits)], stdin=subprocess.PIPE)

    def __del__(self):
        self._subprocess.kill()

    def simulate_sweep(
        self,
        program: Union[circuits.Circuit, schedules.Schedule], # unused
        params: study.Sweepable,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT, # unused
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

        param_dict = {}

        # initial_state: Union[int, np.ndarray],
        if self._initial_state_lockout and initial_state is not None:
            raise Exception(f'Do not supply initial_state in both initialization and simulation.')

        actual_initial_state = 0 if initial_state is None else initial_state
        for target_qubit, initial_value in zip (
            self._qubits,
            # to adhere to Cirq's endian convention:
            [bool(actual_initial_state & (1<<n)) for n in reversed(range(self._num_qubits))]
            ):
            qi0_sym = 'q'+str(target_qubit).zfill(4)+'i0'
            param_dict[ hash(qi0_sym)%(1<<28) ] = '0' if initial_value else '1'
            qi1_sym = 'q'+str(target_qubit).zfill(4)+'i1'
            param_dict[ hash(qi1_sym)%(1<<28) ] = '1' if initial_value else '0'

        param_resolvers = study.to_resolvers(params)
        trial_results = []
        for param_resolver in param_resolvers:

            prep_start = time.time()

            hash_csv = int(hash((self._subprocess, param_resolver, actual_initial_state)))
            self._subprocess.stdin.write(f'cc$H${hash_csv}\n'.encode())

            for hash_key, symbols in self._hash_to_symbols.items():
                param_dict[hash_key] = self._to_java_complex(protocols.resolve_parameters(symbols, param_resolver))

            with open('circuit.lmap', 'r') as lmap_file:
                for line in lmap_file:
                    int_strings = self._re_compile.findall(line)
                    for int_string in int_strings:
                        if int(int_string) in param_dict:
                            line = re.sub(int_string, param_dict[int(int_string)], line)
                    self._subprocess.stdin.write(line.encode())

            print("prep time = ")
            print(time.time() - prep_start)
            java_start = time.time()

            while not os.path.exists(f'{hash_csv}.csv'):
                self._subprocess.stdin.write(b'\n') # keep pushing the BufferedReader

            print("java time = ")
            print(time.time() - java_start)
            post_start = time.time()

            measurements = {}  # type: Dict[str, np.ndarray]
            state_vector = []
            outputQubitString = 0
            with open(f'{hash_csv}.csv', 'r') as csv_file:
                for outputQubitString in range(1<<self._num_qubits):
                    line = csv_file.readline()
                    row = line.split(',')
                    assert int(row[0]) == outputQubitString
                    # measurements[row[0]] = np.array(row[1], dtype=bool)
                    state_vector.append(complex(row[1]))
            assert float(row[2])-1.0 < 1.0/256.0
            os.remove(f'{hash_csv}.csv')

            final_simulator_state = wave_function_simulator.WaveFunctionSimulatorState(qubit_map=self._qubit_map,state_vector=state_vector)
            trial_results.append(
                wave_function_simulator.WaveFunctionTrialResult(
                    params=param_resolver,
                    measurements=measurements,
                    final_simulator_state=final_simulator_state))

            print("post time = ")
            print(time.time() - post_start)

        return trial_results
