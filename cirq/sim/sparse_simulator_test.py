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
from unittest import mock
import numpy as np
import pytest
import sympy

import cirq


def test_invalid_dtype():
    with pytest.raises(ValueError, match='complex'):
        cirq.Simulator(dtype=np.int32)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_no_measurements(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)

    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.X(q1))
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(circuit)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_no_results(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)

    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.X(q1))
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(circuit)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_empty_circuit(dtype):
    simulator = cirq.Simulator(dtype=dtype)
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(cirq.Circuit())


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            result = simulator.run(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]]})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_repetitions_measure_at_end(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    with mock.patch.object(simulator, '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                                (cirq.X**b1)(q1),
                                                cirq.measure(q0),
                                                cirq.measure(q1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements,
                                        {'0': [[b0]] * 3, '1': [[b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count == 4


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_invert_mask_measure_not_terminal(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    with mock.patch.object(simulator,
                           '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit.from_ops(
                    (cirq.X**b0)(q0), (cirq.X**b1)(q1),
                    cirq.measure(q0, q1, key='m', invert_mask=(True, False)),
                    cirq.X(q0))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements,
                                        {'m': [[1 - b0, b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count == 12


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_partial_invert_mask_measure_not_terminal(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    with mock.patch.object(simulator,
                           '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit.from_ops(
                    (cirq.X**b0)(q0), (cirq.X**b1)(q1),
                    cirq.measure(q0, q1, key='m', invert_mask=(True,)),
                    cirq.X(q0))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements,
                                        {'m': [[1 - b0, b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count == 12


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_repetitions_measurement_not_terminal(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    with mock.patch.object(simulator, '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                                (cirq.X**b1)(q1),
                                                cirq.measure(q0),
                                                cirq.measure(q1),
                                                cirq.H(q0),
                                                cirq.H(q1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements,
                                        {'0': [[b0]] * 3, '1': [[b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count == 12


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            param_resolver = cirq.ParamResolver({'b0': b0, 'b1': b1})
            result = simulator.run(circuit, param_resolver=param_resolver)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]] })
            np.testing.assert_equal(result.params, param_resolver)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_mixture(dtype):
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.bit_flip(0.5)(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['0'])[0] < 80
    assert sum(result.measurements['0'])[0] > 20


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_mixture_with_gates(dtype):
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.phase_flip(0.5)(q0),
                                    cirq.H(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['0'])[0] < 80
    assert sum(result.measurements['0'])[0] > 20


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_correlations(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1))
    for _ in range(10):
        result = simulator.run(circuit)
        bits = result.measurements['0,1'][0]
        assert bits[0] == bits[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_measure_multiple_qubits(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0, q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements,
                                    {'0,1': [[b0, b1]] * 3})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_sweeps_param_resolvers(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            params = [cirq.ParamResolver({'b0': b0, 'b1': b1}),
                      cirq.ParamResolver({'b0': b1, 'b1': b0})]
            results = simulator.run_sweep(circuit, params=params)

            assert len(results) == 2
            np.testing.assert_equal(results[0].measurements,
                                    {'0': [[b0]], '1': [[b1]] })
            np.testing.assert_equal(results[1].measurements,
                                    {'0': [[b1]], '1': [[b0]] })
            assert results[0].params == params[0]
            assert results[1].params == params[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_random_unitary(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for _ in range(10):
        random_circuit = cirq.testing.random_circuit(qubits=[q0, q1],
                                                     n_moments=8,
                                                     op_density=0.99)
        circuit_unitary = []
        for x in range(4):
            result = simulator.simulate(random_circuit, qubit_order=[q0, q1],
                                        initial_state=x)
            circuit_unitary.append(result.final_state)
        np.testing.assert_almost_equal(
            np.transpose(circuit_unitary),
            random_circuit.unitary(qubit_order=[q0, q1]),
            decimal=6)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_no_circuit(dtype,):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit()
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([1, 0, 0, 0]))
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate(dtype,):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1))
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([0.5, 0.5, 0.5, 0.5]))
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_mixtures(dtype,):
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=dtype)
    circuit = cirq.Circuit.from_ops(cirq.bit_flip(0.5)(q0), cirq.measure(q0))
    count = 0
    for _ in range(100):
        result = simulator.simulate(circuit, qubit_order=[q0])
        if result.measurements['0']:
            np.testing.assert_almost_equal(result.final_state,
                                            np.array([0, 1]))
            count += 1
        else:
            np.testing.assert_almost_equal(result.final_state,
                                           np.array([1, 0]))
    assert count < 80 and count > 20


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements, {'0': [b0], '1': [b1]})
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][b1] = 1.0
            np.testing.assert_equal(result.final_state,
                                    np.reshape(expected_state, 4))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_initial_state(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            result = simulator.simulate(circuit, initial_state=1)
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][1 - b1] = 1.0
            np.testing.assert_equal(result.final_state,
                                    np.reshape(expected_state, 4))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_qubit_order(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            result = simulator.simulate(circuit, qubit_order=[q1, q0])
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b1][b0] = 1.0
            np.testing.assert_equal(result.final_state,
                                    np.reshape(expected_state, 4))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1))
            resolver = {'b0': b0, 'b1': b1}
            result = simulator.simulate(circuit, param_resolver=resolver)
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][b1] = 1.0
            np.testing.assert_equal(result.final_state,
                                    np.reshape(expected_state, 4))
            assert result.params == cirq.ParamResolver(resolver)
            assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_measure_multiple_qubits(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0, q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0,1': [b0, b1]})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_sweeps_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**sympy.Symbol('b0'))(q0),
                                            (cirq.X**sympy.Symbol('b1'))(q1))
            params = [cirq.ParamResolver({'b0': b0, 'b1': b1}),
                      cirq.ParamResolver({'b0': b1, 'b1': b0})]
            results = simulator.simulate_sweep(circuit, params=params)
            expected_state = np.zeros(shape=(2, 2))
            expected_state[b0][b1] = 1.0
            np.testing.assert_equal(results[0].final_state,
                                    np.reshape(expected_state, 4))

            expected_state = np.zeros(shape=(2, 2))
            expected_state[b1][b0] = 1.0
            np.testing.assert_equal(results[1].final_state,
                                    np.reshape(expected_state, 4))

            assert results[0].params == params[0]
            assert results[1].params == params[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1), cirq.H(q0),
                                    cirq.H(q1))
    simulator = cirq.Simulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            np.testing.assert_almost_equal(step.state_vector(),
                                           np.array([0.5] * 4))
        else:
            np.testing.assert_almost_equal(step.state_vector(),
                                           np.array([1, 0, 0, 0]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_empty_circuit(dtype):
    circuit = cirq.Circuit()
    simulator = cirq.Simulator(dtype=dtype)
    step = None
    for step in simulator.simulate_moment_steps(circuit):
        pass
    assert step._simulator_state() == cirq.WaveFunctionSimulatorState(
        state_vector=np.array([1]), qubit_map={})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_set_state(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.H(q1), cirq.H(q0),
                                    cirq.H(q1))
    simulator = cirq.Simulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        np.testing.assert_almost_equal(step.state_vector(), np.array([0.5] * 4))
        if i == 0:
            step.set_state_vector(np.array([1, 0, 0, 0], dtype=dtype))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_sample(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = cirq.Simulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, False])
                        or np.array_equal(sample, [False, False]))
        else:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, True])
                        or np.array_equal(sample, [False, False]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_intermediate_measurement(dtype):
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.measure(q0), cirq.H(q0))
    simulator = cirq.Simulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 1:
            result = int(step.measurements['0'][0])
            expected = np.zeros(2)
            expected[result] = 1
            np.testing.assert_almost_equal(step.state_vector(), expected)
        if i == 2:
            expected = np.array([np.sqrt(0.5), np.sqrt(0.5) * (-1) ** result])
            np.testing.assert_almost_equal(step.state_vector(), expected)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_compute_displays(dtype):
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit.from_ops(
        cirq.pauli_string_expectation(
            cirq.PauliString({qubits[3]: cirq.Z}),
            key='z3'
        ),
        cirq.X(qubits[1]),
        cirq.pauli_string_expectation(
            cirq.PauliString({qubits[0]: cirq.Z,
                              qubits[1]: cirq.Z}),
            key='z0z1'
        ),
        cirq.pauli_string_expectation(
            cirq.PauliString({qubits[0]: cirq.Z,
                              qubits[1]: cirq.X}),
            key='z0x1'
        ),
        cirq.H(qubits[2]),
        cirq.X(qubits[3]),
        cirq.H(qubits[3]),
        cirq.pauli_string_expectation(
            cirq.PauliString({qubits[1]: cirq.Z,
                              qubits[2]: cirq.X}),
            key='z1x2'
        ),
        cirq.pauli_string_expectation(
            cirq.PauliString({qubits[0]: cirq.X,
                              qubits[1]: cirq.Z}),
            key='x0z1'
        ),
        cirq.pauli_string_expectation(
            cirq.PauliString({qubits[3]: cirq.X}),
            key='x3'
        ),
        cirq.pauli_string_expectation(
            cirq.PauliString({qubits[1]: cirq.Z,
                              qubits[2]: cirq.X}),
            num_samples=1,
            key='approx_z1x2'
        ),
    )
    simulator = cirq.Simulator(dtype=dtype)
    result = simulator.compute_displays(circuit)

    np.testing.assert_allclose(result.display_values['z3'], 1, atol=1e-7)
    np.testing.assert_allclose(result.display_values['z0z1'], -1, atol=1e-7)
    np.testing.assert_allclose(result.display_values['z0x1'], 0, atol=1e-7)
    np.testing.assert_allclose(result.display_values['z1x2'], -1, atol=1e-7)
    np.testing.assert_allclose(result.display_values['x0z1'], 0, atol=1e-7)
    np.testing.assert_allclose(result.display_values['x3'], -1, atol=1e-7)
    np.testing.assert_allclose(result.display_values['approx_z1x2'], -1,
                               atol=1e-7)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_compute_samples_displays(dtype):
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(
        cirq.X(a),
        cirq.H(b),
        cirq.X(c),
        cirq.H(c),
        cirq.pauli_string_expectation(
            cirq.PauliString({c: cirq.X}),
            key='x3'
        ),
        cirq.pauli_string_expectation(
            cirq.PauliString({a: cirq.Z,
                              b: cirq.X}),
            num_samples=10,
            key='approx_z1x2'
        ),
        cirq.pauli_string_expectation(
            cirq.PauliString({a: cirq.Z,
                              c: cirq.X}),
            num_samples=10,
            key='approx_z1x3'
        ),
    )
    simulator = cirq.Simulator(dtype=dtype)
    result = simulator.compute_samples_displays(circuit)

    assert 'x3' not in result.display_values
    np.testing.assert_allclose(result.display_values['approx_z1x2'], -1,
                               atol=1e-7)
    np.testing.assert_allclose(result.display_values['approx_z1x3'], 1,
                               atol=1e-7)


def test_invalid_run_no_unitary():
    class NoUnitary(cirq.SingleQubitGate):
        pass
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator()
    circuit = cirq.Circuit.from_ops(NoUnitary()(q0))
    circuit.append([cirq.measure(q0, key='meas')])
    with pytest.raises(TypeError, match='unitary'):
        simulator.run(circuit)


def test_allocates_new_state():
    class NoUnitary(cirq.SingleQubitGate):

        def _has_unitary_(self):
            return True

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return np.copy(args.target_tensor)

    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator()
    circuit = cirq.Circuit.from_ops(NoUnitary()(q0))

    initial_state = np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.complex64)
    result = simulator.simulate(circuit, initial_state=initial_state)
    np.testing.assert_array_almost_equal(result.state_vector(), initial_state)
    assert not initial_state is result.state_vector()


def test_simulator_step_state_mixin():
    qubits = cirq.LineQubit.range(2)
    qubit_map = {qubits[i]: i for i in range(2)}
    result = cirq.SparseSimulatorStep(
        measurements={'m': np.array([1, 2])},
        state_vector=np.array([0, 1, 0, 0]),
        qubit_map=qubit_map,
        dtype=np.complex64)
    rho = np.array([[0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho,
                                         result.density_matrix_of(qubits))
    bloch = np.array([0,0,-1])
    np.testing.assert_array_almost_equal(bloch,
                                         result.bloch_vector_of(qubits[1]))

    assert result.dirac_notation() == '|01⟩'


class MultiHTestGate(cirq.TwoQubitGate):
    def _decompose_(self, qubits):
        return cirq.H.on_each(*qubits)


def test_simulates_composite():
    c = cirq.Circuit.from_ops(MultiHTestGate().on(*cirq.LineQubit.range(2)))
    expected = np.array([0.5] * 4)
    np.testing.assert_allclose(c.final_wavefunction(), expected)
    np.testing.assert_allclose(cirq.Simulator().simulate(c).state_vector(),
                               expected)


def test_simulate_measurement_inversions():
    q = cirq.NamedQubit('q')

    c = cirq.Circuit.from_ops(cirq.measure(q, key='q', invert_mask=(True,)))
    assert cirq.Simulator().simulate(c).measurements == {'q': np.array([True])}

    c = cirq.Circuit.from_ops(cirq.measure(q, key='q', invert_mask=(False,)))
    assert cirq.Simulator().simulate(c).measurements == {'q': np.array([False])}


def test_works_on_pauli_string_phasor():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit.from_ops(np.exp(1j * np.pi * cirq.X(a) * cirq.X(b)))
    sim = cirq.Simulator()
    result = sim.simulate(c).state_vector()
    np.testing.assert_allclose(result.reshape(4),
                               np.array([0, 0, 0, 1j]),
                               atol=1e-8)


def test_works_on_pauli_string():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit.from_ops(cirq.X(a) * cirq.X(b))
    sim = cirq.Simulator()
    result = sim.simulate(c).state_vector()
    np.testing.assert_allclose(result.reshape(4),
                               np.array([0, 0, 0, 1]),
                               atol=1e-8)


def test_measure_at_end_invert_mask():
    simulator = cirq.Simulator()
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit.from_ops(
        cirq.measure(a, key='a', invert_mask=(True,)))
    result = simulator.run(circuit, repetitions=4)
    np.testing.assert_equal(result.measurements['a'], np.array([[1]] * 4))


def test_measure_at_end_invert_mask_multiple_qubits():
    simulator = cirq.Simulator()
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(
        cirq.measure(a, key='a', invert_mask=(True,)),
        cirq.measure(b, c, key='bc', invert_mask=(False, True)))
    result = simulator.run(circuit, repetitions=4)
    np.testing.assert_equal(result.measurements['a'], np.array([[True]] * 4))
    np.testing.assert_equal(result.measurements['bc'], np.array([[0, 1]] * 4))


def test_measure_at_end_invert_mask_partial():
    simulator = cirq.Simulator()
    a, _, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(
        cirq.measure(a, c, key='ac', invert_mask=(True,)))
    result = simulator.run(circuit, repetitions=4)
    np.testing.assert_equal(result.measurements['ac'], np.array([[1, 0]] * 4))


def test_compute_amplitudes():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit.from_ops(cirq.X(a), cirq.H(a), cirq.H(b))
    sim = cirq.Simulator()

    result = sim.compute_amplitudes(c, np.array([[0, 0]]))
    np.testing.assert_allclose(np.array(result), np.array([0.5]))

    result = sim.compute_amplitudes(c, np.array([[0, 1], [1, 0], [1, 1]]))
    np.testing.assert_allclose(np.array(result), np.array([0.5, -0.5, -0.5]))

    result = sim.compute_amplitudes(c,
                                    np.array([[0, 1], [1, 0], [1, 1]]),
                                    qubit_order=(b, a))
    np.testing.assert_allclose(np.array(result), np.array([-0.5, 0.5, -0.5]))


def test_run_sweep_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = cirq.Simulator()
    circuit = cirq.Circuit.from_ops(
        cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.run_sweep(circuit, cirq.ParamResolver({}))


def test_simulate_sweep_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = cirq.Simulator()
    circuit = cirq.Circuit.from_ops(
        cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.simulate_sweep(circuit, cirq.ParamResolver({}))


def test_random_seed():
    sim = cirq.Simulator(seed=1234)
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit.from_ops(cirq.X(a)**0.5, cirq.measure(a))
    result = sim.run(circuit, repetitions=10)
    print(result.measurements['a'])
    assert np.all(
        result.measurements['a'] == [[False], [True], [False], [True], [True],
                                     [False], [False], [True], [True], [True]])
