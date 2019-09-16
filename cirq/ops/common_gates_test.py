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

import itertools

import numpy as np
import pytest
import sympy

import cirq


H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = cirq.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1],
                 [1, 1j, -1, -1j],
                 [1, -1, 1, -1],
                 [1, -1j, -1, 1j]]) * 0.5


@pytest.mark.parametrize('eigen_gate_type', [
    cirq.CZPowGate,
    cirq.XPowGate,
    cirq.YPowGate,
    cirq.ZPowGate,
])
def test_phase_insensitive_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(
            eigen_gate_type)


@pytest.mark.parametrize('eigen_gate_type', [
    cirq.CNotPowGate,
    cirq.HPowGate,
    cirq.ISwapPowGate,
    cirq.SwapPowGate,
])
def test_phase_sensitive_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(
            eigen_gate_type, ignoring_global_phase=True)

@pytest.mark.parametrize('gate_type, num_qubits',
    itertools.product(
        (cirq.MeasurementGate, cirq.IdentityGate),
        range(1, 5))
)
def test_consistent_protocols(gate_type, num_qubits):
    gate = gate_type(num_qubits=num_qubits)
    cirq.testing.assert_implements_consistent_protocols(
        gate, qubit_count=num_qubits)

    gate = gate_type(num_qubits=num_qubits, qid_shape=(3,) * num_qubits)
    cirq.testing.assert_implements_consistent_protocols(gate,
                                                        qubit_count=num_qubits)


def test_cz_init():
    assert cirq.CZPowGate(exponent=0.5).exponent == 0.5
    assert cirq.CZPowGate(exponent=5).exponent == 5
    assert (cirq.CZ**0.5).exponent == 0.5


def test_transformations():
    pi = np.pi
    initialRx = cirq.Rx(0.4)
    expectedPowx = cirq.X**(0.4 / pi)
    receivedPowx = initialRx.with_canonical_global_phase()
    backToRx = receivedPowx.in_su2()
    assert receivedPowx == expectedPowx
    assert backToRx == initialRx
    initialRy = cirq.Ry(0.123)
    expectedPowy = cirq.Y**(0.123 / pi)
    receivedPowy = initialRy.with_canonical_global_phase()
    backToRy = receivedPowy.in_su2()
    assert receivedPowy == expectedPowy
    assert backToRy == initialRy
    initialRz = cirq.Rz(-1.53)
    expectedPowz = cirq.Z**(-1.53 / pi)
    receivedPowz = initialRz.with_canonical_global_phase()
    backToRz = receivedPowz.in_su2()
    assert receivedPowz == expectedPowz
    assert backToRz == initialRz


def test_cz_str():
    assert str(cirq.CZ) == 'CZ'
    assert str(cirq.CZ**0.5) == 'CZ**0.5'
    assert str(cirq.CZ**-0.25) == 'CZ**-0.25'


def test_cz_repr():
    assert repr(cirq.CZ) == 'cirq.CZ'
    assert repr(cirq.CZ**0.5) == '(cirq.CZ**0.5)'
    assert repr(cirq.CZ**-0.25) == '(cirq.CZ**-0.25)'


def test_cz_unitary():
    assert np.allclose(cirq.unitary(cirq.CZ),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]))

    assert np.allclose(cirq.unitary(cirq.CZ**0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1j]]))

    assert np.allclose(cirq.unitary(cirq.CZ**0),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ**-0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1j]]))


def test_z_init():
    z = cirq.ZPowGate(exponent=5)
    assert z.exponent == 5

    # Canonicalizes exponent for equality, but keeps the inner details.
    assert cirq.Z**0.5 != cirq.Z**-0.5
    assert (cirq.Z**-1)**0.5 == cirq.Z**-0.5
    assert cirq.Z**-1 == cirq.Z


def test_rot_gates_eq():
    eq = cirq.testing.EqualsTester()
    gates = [
        lambda p: cirq.CZ**p,
        lambda p: cirq.X**p,
        lambda p: cirq.Y**p,
        lambda p: cirq.Z**p,
        lambda p: cirq.CNOT**p,
    ]
    for gate in gates:
        eq.add_equality_group(gate(3.5),
                              gate(-0.5))
        eq.make_equality_group(lambda: gate(0))
        eq.make_equality_group(lambda: gate(0.5))

    eq.add_equality_group(cirq.XPowGate(), cirq.XPowGate(exponent=1), cirq.X)
    eq.add_equality_group(cirq.YPowGate(), cirq.YPowGate(exponent=1), cirq.Y)
    eq.add_equality_group(cirq.ZPowGate(), cirq.ZPowGate(exponent=1), cirq.Z)
    eq.add_equality_group(cirq.ZPowGate(exponent=1,
                                        global_shift=-0.5),
                          cirq.ZPowGate(exponent=5,
                                        global_shift=-0.5))
    eq.add_equality_group(cirq.ZPowGate(exponent=3,
                                        global_shift=-0.5))
    eq.add_equality_group(cirq.ZPowGate(exponent=1,
                                        global_shift=-0.1))
    eq.add_equality_group(cirq.ZPowGate(exponent=5,
                                        global_shift=-0.1))
    eq.add_equality_group(cirq.CNotPowGate(),
                          cirq.CNotPowGate(exponent=1),
                          cirq.CNOT)
    eq.add_equality_group(cirq.CZPowGate(),
                          cirq.CZPowGate(exponent=1), cirq.CZ)


def test_z_unitary():
    assert np.allclose(cirq.unitary(cirq.Z),
                       np.array([[1, 0], [0, -1]]))
    assert np.allclose(cirq.unitary(cirq.Z**0.5),
                       np.array([[1, 0], [0, 1j]]))
    assert np.allclose(cirq.unitary(cirq.Z**0),
                       np.array([[1, 0], [0, 1]]))
    assert np.allclose(cirq.unitary(cirq.Z**-0.5),
                       np.array([[1, 0], [0, -1j]]))


def test_y_unitary():
    assert np.allclose(cirq.unitary(cirq.Y),
                       np.array([[0, -1j], [1j, 0]]))

    assert np.allclose(cirq.unitary(cirq.Y**0.5),
                       np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2)

    assert np.allclose(cirq.unitary(cirq.Y**0),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(cirq.unitary(cirq.Y**-0.5),
                       np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2)


def test_x_unitary():
    assert np.allclose(cirq.unitary(cirq.X),
                       np.array([[0, 1], [1, 0]]))

    assert np.allclose(cirq.unitary(cirq.X**0.5),
                       np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)

    assert np.allclose(cirq.unitary(cirq.X**0),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(cirq.unitary(cirq.X**-0.5),
                       np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)


@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_identity_init(num_qubits):
    assert cirq.IdentityGate(num_qubits).num_qubits() == num_qubits
    assert cirq.qid_shape(cirq.IdentityGate(num_qubits)) == (2,) * num_qubits
    assert cirq.qid_shape(cirq.IdentityGate(3, (1, 2, 3))) == (1, 2, 3)
    assert cirq.qid_shape(cirq.IdentityGate(qid_shape=(1, 2, 3))) == (1, 2, 3)
    with pytest.raises(ValueError, match='len.* !='):
        cirq.IdentityGate(5, qid_shape=(1, 2))
    with pytest.raises(ValueError, match='Specify either'):
        cirq.IdentityGate()


def test_identity_on_each():
    q0, q1, q2 = cirq.LineQubit.range(3)
    assert cirq.I.on_each(q0, q1, q2) == [cirq.I(q0), cirq.I(q1), cirq.I(q2)]
    assert cirq.I.on_each([q0, [q1],
                           q2]) == [cirq.I(q0),
                                    cirq.I(q1),
                                    cirq.I(q2)]
    assert cirq.I.on_each(iter([q0, [q1],
                                q2])) == [cirq.I(q0),
                                          cirq.I(q1),
                                          cirq.I(q2)]
    with pytest.raises(ValueError, match='str'):
        cirq.I.on_each('abc')


def test_identity_on_each_only_single_qubit():
    q0, q1 = cirq.LineQubit.range(2)
    q0_3, q1_3 = q0.with_dimension(3), q1.with_dimension(3)
    assert cirq.I.on_each(q0, q1) == [cirq.I.on(q0), cirq.I.on(q1)]
    assert cirq.IdentityGate(1, (3,)).on_each(q0_3, q1_3) == [
        cirq.IdentityGate(1, (3,)).on(q0_3),
        cirq.IdentityGate(1, (3,)).on(q1_3),
    ]
    with pytest.raises(ValueError, match='one qubit'):
        cirq.IdentityGate(num_qubits=2).on_each(q0, q1)


@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_identity_unitary(num_qubits):
    i = cirq.IdentityGate(num_qubits)
    assert np.allclose(cirq.unitary(i), np.identity(2 ** num_qubits))
    i3 = cirq.IdentityGate(num_qubits, (3,) * num_qubits)
    assert np.allclose(cirq.unitary(i3), np.identity(3**num_qubits))


def test_identity_str():
    assert str(cirq.IdentityGate(1)) == 'I'
    assert str(cirq.IdentityGate(2)) == 'I(2)'
    # Qid shape is not included in str
    assert str(cirq.IdentityGate(1, (3,))) == 'I'
    assert str(cirq.IdentityGate(2, (1, 2))) == 'I(2)'


def test_identity_repr():
    assert repr(cirq.IdentityGate(2)) == 'cirq.IdentityGate(2)'
    assert repr(cirq.IdentityGate(2, (2, 3))) == 'cirq.IdentityGate(2, (2, 3))'


def test_identity_apply_unitary():
    v = np.array([1, 0])
    result = cirq.apply_unitary(
        cirq.I, cirq.ApplyUnitaryArgs(v, np.array([0, 1]), (0,)))
    assert result is v

    v = np.array([1, 0, 0])
    result = cirq.apply_unitary(
        cirq.IdentityGate(1, (3,)),
        cirq.ApplyUnitaryArgs(v, np.array([0, 1, 2]), (0,)))
    assert result is v


def test_identity_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.make_equality_group(
        lambda: cirq.I,
        lambda: cirq.IdentityGate(1),
        lambda: cirq.IdentityGate(1, (2,)),
    )
    equals_tester.add_equality_group(cirq.IdentityGate(2),
                                     cirq.IdentityGate(2, (2, 2)))
    equals_tester.add_equality_group(cirq.IdentityGate(4))
    equals_tester.add_equality_group(cirq.IdentityGate(1, (3,)))
    equals_tester.add_equality_group(cirq.IdentityGate(4, (1, 2, 3, 4)))


def test_identity_global():
    qubits = cirq.LineQubit.range(3)
    assert cirq.identity(*qubits) == cirq.IdentityGate(3).on(*qubits)
    qids = cirq.LineQid.for_qid_shape((1, 2, 3))
    assert cirq.identity(*qids) == cirq.IdentityGate(3, (1, 2, 3)).on(*qids)
    with pytest.raises(ValueError, match='type different'):
        cirq.identity(qubits)  # The user forgot to expand the list for example.


def test_h_unitary():
    sqrt = cirq.unitary(cirq.H**0.5)
    m = np.dot(sqrt, sqrt)
    assert np.allclose(m, cirq.unitary(cirq.H), atol=1e-8)


def test_h_init():
    h = cirq.HPowGate(exponent=0.5)
    assert h.exponent == 0.5


def test_h_str():
    assert str(cirq.H) == 'H'
    assert str(cirq.H**0.5) == 'H^0.5'


def test_runtime_types_of_rot_gates():
    for gate_type in [lambda p: cirq.CZPowGate(exponent=p),
                      lambda p: cirq.XPowGate(exponent=p),
                      lambda p: cirq.YPowGate(exponent=p),
                      lambda p: cirq.ZPowGate(exponent=p)]:
        p = gate_type(sympy.Symbol('a'))
        assert cirq.unitary(p, None) is None
        assert cirq.pow(p, 2, None) == gate_type(2 * sympy.Symbol('a'))
        assert cirq.inverse(p, None) == gate_type(-sympy.Symbol('a'))

        c = gate_type(0.5)
        assert cirq.unitary(c, None) is not None
        assert cirq.pow(c, 2) == gate_type(1)
        assert cirq.inverse(c) == gate_type(-0.5)


@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_measure_init(num_qubits):
    assert cirq.MeasurementGate(num_qubits).num_qubits() == num_qubits
    assert cirq.MeasurementGate(num_qubits, key='a').key == 'a'
    assert cirq.MeasurementGate(num_qubits,
                                invert_mask=(True,)).invert_mask == (True,)
    assert cirq.qid_shape(cirq.MeasurementGate(num_qubits)) == (2,) * num_qubits
    assert cirq.qid_shape(cirq.MeasurementGate(3, qid_shape=(1, 2,
                                                             3))) == (1, 2, 3)
    assert cirq.qid_shape(cirq.MeasurementGate(qid_shape=(1, 2, 3))) == (1, 2,
                                                                         3)
    with pytest.raises(ValueError, match='len.* >'):
        cirq.MeasurementGate(5, invert_mask=(True,) * 6)
    with pytest.raises(ValueError, match='len.* !='):
        cirq.MeasurementGate(5, qid_shape=(1, 2))
    with pytest.raises(ValueError, match='Specify either'):
        cirq.MeasurementGate()


def test_measurement_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.MeasurementGate(
        1, ''), lambda: cirq.MeasurementGate(1, '', invert_mask=()), lambda:
                           cirq.MeasurementGate(1, '', qid_shape=(2,)))
    eq.add_equality_group(cirq.MeasurementGate(1, 'a'))
    eq.add_equality_group(cirq.MeasurementGate(1, 'a', invert_mask=(True,)))
    eq.add_equality_group(cirq.MeasurementGate(1, 'a', invert_mask=(False,)))
    eq.add_equality_group(cirq.MeasurementGate(1, 'b'))
    eq.add_equality_group(cirq.MeasurementGate(2, 'a'))
    eq.add_equality_group(cirq.MeasurementGate(2, ''))
    eq.add_equality_group(cirq.MeasurementGate(3, 'a'),
                          cirq.MeasurementGate(3, 'a', qid_shape=(2, 2, 2)))
    eq.add_equality_group(cirq.MeasurementGate(3, 'a', qid_shape=(1, 2, 3)))


def test_measurement_full_invert_mask():
    assert cirq.MeasurementGate(1, 'a').full_invert_mask() == (False,)
    assert (cirq.MeasurementGate(
        2, 'a', invert_mask=(False, True)).full_invert_mask() == (False, True))
    assert (cirq.MeasurementGate(
        2, 'a', invert_mask=(True,)).full_invert_mask() == (True, False))


def test_qudit_measure_qasm():
    assert cirq.qasm(cirq.measure(cirq.LineQid(0, 3), key='a'),
                     args=cirq.QasmArgs(),
                     default='not implemented') == 'not implemented'


def test_interchangeable_qubit_eq():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(cirq.SWAP(a, b), cirq.SWAP(b, a))
    eq.add_equality_group(cirq.SWAP(a, c))

    eq.add_equality_group(cirq.CZ(a, b), cirq.CZ(b, a))
    eq.add_equality_group(cirq.CZ(a, c))

    eq.add_equality_group(cirq.CNOT(a, b))
    eq.add_equality_group(cirq.CNOT(b, a))
    eq.add_equality_group(cirq.CNOT(a, c))


def test_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(
        cirq.SWAP(a, b),
        cirq.X(a),
        cirq.Y(a),
        cirq.Z(a),
        cirq.Z(a)**sympy.Symbol('x'),
        cirq.Rx(sympy.Symbol('x')).on(a),
        cirq.CZ(a, b),
        cirq.CNOT(a, b),
        cirq.CNOT(b, a),
        cirq.H(a)**0.5,
        cirq.ISWAP(a, b)**-1,
        cirq.I(a),
        cirq.IdentityGate(2)(a, b))

    cirq.testing.assert_has_diagram(circuit, """
a: ───×───X───Y───Z───Z^x───Rx(x)───@───@───X───H^0.5───iSwap──────I───I───
      │                             │   │   │           │              │
b: ───×─────────────────────────────@───X───@───────────iSwap^-1───────I───
""")

    cirq.testing.assert_has_diagram(circuit, """
a: ---swap---X---Y---Z---Z^x---Rx(x)---@---@---X---H^0.5---iSwap------I---I---
      |                                |   |   |           |              |
b: ---swap-----------------------------@---X---@-----------iSwap^-1-------I---
""", use_unicode_characters=False)


def test_cnot_unitary():
    np.testing.assert_almost_equal(
        cirq.unitary(cirq.CNOT**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5+0.5j, 0.5-0.5j],
            [0, 0, 0.5-0.5j, 0.5+0.5j],
        ]))


def test_cnot_keyword_arguments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(cirq.CNOT(a, b),
                                 cirq.CNOT(control=a, target=b))
    eq_tester.add_equality_group(cirq.CNOT(b, a),
                                 cirq.CNOT(control=b, target=a))


def test_cnot_keyword_not_equal():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(AssertionError):
        eq_tester = cirq.testing.EqualsTester()
        eq_tester.add_equality_group(cirq.CNOT(a, b),
                                     cirq.CNOT(target=a, control=b))


def test_cnot_keyword_too_few_arguments():
    a = cirq.NamedQubit('a')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(control=a)


def test_cnot_mixed_keyword_and_positional_arguments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(a, target=b)


def test_cnot_unknown_keyword_argument():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(target=a, controlled=b)


def test_cnot_decompose():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.decompose_once(cirq.CNOT(a, b)**sympy.Symbol('x')) is not None


def test_swap_unitary():
    np.testing.assert_almost_equal(
        cirq.unitary(cirq.SWAP**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
            [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
            [0, 0, 0, 1]
        ]))


def test_repr():
    assert repr(cirq.X) == 'cirq.X'
    assert repr(cirq.X**0.5) == '(cirq.X**0.5)'

    assert repr(cirq.Z) == 'cirq.Z'
    assert repr(cirq.Z**0.5) == 'cirq.S'
    assert repr(cirq.Z**0.25) == 'cirq.T'
    assert repr(cirq.Z**0.125) == '(cirq.Z**0.125)'

    assert repr(cirq.S) == 'cirq.S'
    assert repr(cirq.S**-1) == '(cirq.S**-1)'
    assert repr(cirq.T) == 'cirq.T'
    assert repr(cirq.T**-1) == '(cirq.T**-1)'

    assert repr(cirq.Y) == 'cirq.Y'
    assert repr(cirq.Y**0.5) == '(cirq.Y**0.5)'

    assert repr(cirq.I) == 'cirq.I'

    assert repr(cirq.CNOT) == 'cirq.CNOT'
    assert repr(cirq.CNOT**0.5) == '(cirq.CNOT**0.5)'

    assert repr(cirq.SWAP) == 'cirq.SWAP'
    assert repr(cirq.SWAP ** 0.5) == '(cirq.SWAP**0.5)'

    assert repr(cirq.ISWAP) == 'cirq.ISWAP'
    assert repr(cirq.ISWAP ** 0.5) == '(cirq.ISWAP**0.5)'

    cirq.testing.assert_equivalent_repr(
        cirq.X**(sympy.Symbol('a') / 2 - sympy.Symbol('c') * 3 + 5))

    # There should be no floating point error during initialization, and repr
    # should be using the "shortest decimal value closer to X than any other
    # floating point value" strategy, as opposed to the "exactly value in
    # decimal" strategy.
    assert repr(cirq.CZ**0.2) == '(cirq.CZ**0.2)'


def test_str():
    assert str(cirq.X) == 'X'
    assert str(cirq.X**0.5) == 'X**0.5'
    assert str(cirq.Rx(np.pi)) == 'Rx(π)'
    assert str(cirq.Rx(0.5 * np.pi)) == 'Rx(0.5π)'
    assert str(cirq.XPowGate(
        global_shift=-0.25)) == 'XPowGate(exponent=1.0, global_shift=-0.25)'

    assert str(cirq.Z) == 'Z'
    assert str(cirq.Z**0.5) == 'S'
    assert str(cirq.Z**0.125) == 'Z**0.125'
    assert str(cirq.Rz(np.pi)) == 'Rz(π)'
    assert str(cirq.Rz(1.4 * np.pi)) == 'Rz(1.4π)'
    assert str(cirq.ZPowGate(
        global_shift=0.25)) == 'ZPowGate(exponent=1.0, global_shift=0.25)'

    assert str(cirq.S) == 'S'
    assert str(cirq.S**-1) == 'S**-1'
    assert str(cirq.T) == 'T'
    assert str(cirq.T**-1) == 'T**-1'

    assert str(cirq.Y) == 'Y'
    assert str(cirq.Y**0.5) == 'Y**0.5'
    assert str(cirq.Ry(np.pi)) == 'Ry(π)'
    assert str(cirq.Ry(3.14 * np.pi)) == 'Ry(3.14π)'
    assert str(cirq.YPowGate(
        exponent=2,
        global_shift=-0.25)) == 'YPowGate(exponent=2, global_shift=-0.25)'

    assert str(cirq.CX) == 'CNOT'
    assert str(cirq.CNOT**0.5) == 'CNOT**0.5'

    assert str(cirq.SWAP) == 'SWAP'
    assert str(cirq.SWAP**0.5) == 'SWAP**0.5'

    assert str(cirq.ISWAP) == 'ISWAP'
    assert str(cirq.ISWAP**0.5) == 'ISWAP**0.5'

def test_measurement_gate_diagram():
    # Shows key.
    assert cirq.circuit_diagram_info(cirq.MeasurementGate(1)
                                     ) == cirq.CircuitDiagramInfo(("M('')",))
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(1, key='test')
    ) == cirq.CircuitDiagramInfo(("M('test')",))

    # Uses known qubit count.
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(3),
        cirq.CircuitDiagramInfoArgs(
            known_qubits=None,
            known_qubit_count=3,
            use_unicode_characters=True,
            precision=None,
            qubit_map=None
        )) == cirq.CircuitDiagramInfo(("M('')", 'M', 'M'))

    # Shows invert mask.
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(2, invert_mask=(False, True))
    ) == cirq.CircuitDiagramInfo(("M('')", "!M"))

    # Omits key when it is the default.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b)), """
a: ───M───
      │
b: ───M───
""")
    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b, invert_mask=(True,))), """
a: ───!M───
      │
b: ───M────
""")
    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b, key='test')), """
a: ───M('test')───
      │
b: ───M───────────
""")


def test_measure():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Empty application.
    with pytest.raises(ValueError, match='empty set of qubits'):
        _ = cirq.measure()

    assert cirq.measure(a) == cirq.MeasurementGate(num_qubits=1, key='a').on(a)
    assert cirq.measure(a, b) == cirq.MeasurementGate(num_qubits=2,
                                                      key='a,b').on(a, b)
    assert cirq.measure(b, a) == cirq.MeasurementGate(num_qubits=2,
                                                      key='b,a').on(b, a)
    assert cirq.measure(a, key='b') == cirq.MeasurementGate(num_qubits=1,
                                                            key='b').on(a)
    assert cirq.measure(a, invert_mask=(True,)) == cirq.MeasurementGate(
        num_qubits=1, key='a', invert_mask=(True,)).on(a)
    assert cirq.measure(*cirq.LineQid.for_qid_shape(
        (1, 2, 3)), key='a') == cirq.MeasurementGate(
            num_qubits=3, key='a',
            qid_shape=(1, 2, 3)).on(*cirq.LineQid.for_qid_shape((1, 2, 3)))

    with pytest.raises(ValueError, match='ndarray'):
        _ = cirq.measure(np.ndarray([1, 0]))

    with pytest.raises(ValueError, match='Qid'):
        _ = cirq.measure("bork")


def test_measurement_channel():
    np.testing.assert_allclose(
            cirq.channel(cirq.MeasurementGate(1)),
            (np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])))
    # yapf: disable
    np.testing.assert_allclose(
            cirq.channel(cirq.MeasurementGate(2)),
            (np.array([[1, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 1]])))
    np.testing.assert_allclose(
            cirq.channel(cirq.MeasurementGate(2, qid_shape=(2, 3))),
            (np.diag([1, 0, 0, 0, 0, 0]),
             np.diag([0, 1, 0, 0, 0, 0]),
             np.diag([0, 0, 1, 0, 0, 0]),
             np.diag([0, 0, 0, 1, 0, 0]),
             np.diag([0, 0, 0, 0, 1, 0]),
             np.diag([0, 0, 0, 0, 0, 1])))
    # yapf: enable


def test_measurement_qubit_count_vs_mask_length():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    _ = cirq.MeasurementGate(num_qubits=1, invert_mask=(True,)).on(a)
    _ = cirq.MeasurementGate(num_qubits=2, invert_mask=(True, False)).on(a, b)
    _ = cirq.MeasurementGate(num_qubits=3, invert_mask=(True, False, True)).on(
        a, b, c)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(num_qubits=1, invert_mask=(True, False)).on(a)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(num_qubits=3,
                                 invert_mask=(True, False, True)).on(a, b)


def test_measure_each():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.measure_each() == []
    assert cirq.measure_each(a) == [cirq.measure(a)]
    assert cirq.measure_each(a, b) == [cirq.measure(a), cirq.measure(b)]
    assert cirq.measure_each(a.with_dimension(3), b.with_dimension(3)) == [
        cirq.measure(a.with_dimension(3)),
        cirq.measure(b.with_dimension(3))
    ]

    assert cirq.measure_each(a, b, key_func=lambda e: e.name + '!') == [
        cirq.measure(a, key='a!'),
        cirq.measure(b, key='b!')
    ]


def test_iswap_str():
    assert str(cirq.ISWAP) == 'ISWAP'
    assert str(cirq.ISWAP**0.5) == 'ISWAP**0.5'


def test_iswap_unitary():
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.ISWAP),
        # Reference for the iswap gate's matrix using +i instead of -i:
        # https://quantumcomputing.stackexchange.com/questions/2594/
        np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
        atol=1e-8)


def test_iswap_decompose_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    decomposed = cirq.Circuit.from_ops(
        cirq.decompose_once(cirq.ISWAP(a, b)**0.5))
    cirq.testing.assert_has_diagram(decomposed, """
a: ───@───H───X───T───X───T^-1───H───@───
      │       │       │              │
b: ───X───────@───────@──────────────X───
""")


def test_rx_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(np.pi / 2)),
        np.array([[s, -s*1j], [-s*1j, s]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(-np.pi / 2)),
        np.array([[s, s*1j], [s*1j, s]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(0)),
        np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(2 * np.pi)),
        np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(np.pi)),
        np.array([[0, -1j], [-1j, 0]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(-np.pi)),
        np.array([[0, 1j], [1j, 0]]))


def test_ry_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(np.pi / 2)),
        np.array([[s, -s], [s, s]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(-np.pi / 2)),
        np.array([[s, s], [-s, s]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(0)),
        np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(2 * np.pi)),
        np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(np.pi)),
        np.array([[0, -1], [1, 0]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(-np.pi)),
        np.array([[0, 1], [-1, 0]]))


def test_rz_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(np.pi / 2)),
        np.array([[s - s*1j, 0], [0, s + s*1j]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(-np.pi / 2)),
        np.array([[s + s*1j, 0], [0, s - s*1j]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(0)),
        np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(2 * np.pi)),
        np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(np.pi)),
        np.array([[-1j, 0], [0, 1j]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(-np.pi)),
        np.array([[1j, 0], [0, -1j]]))


def test_phase_by_xy():
    assert cirq.phase_by(cirq.X, 0.25, 0) == cirq.Y
    assert cirq.phase_by(cirq.X**0.5, 0.25, 0) == cirq.Y**0.5
    assert cirq.phase_by(cirq.X**-0.5, 0.25, 0) == cirq.Y**-0.5


def test_ixyz_circuit_diagram():
    q = cirq.NamedQubit('q')
    ix = cirq.XPowGate(exponent=1, global_shift=0.5)
    iy = cirq.YPowGate(exponent=1, global_shift=0.5)
    iz = cirq.ZPowGate(exponent=1, global_shift=0.5)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            ix(q),
            ix(q)**-1,
            ix(q)**-0.99999,
            ix(q)**-1.00001,
            ix(q)**3,
            ix(q)**4.5,
            ix(q)**4.500001,
        ), """
q: ───X───X───X───X───X───X^0.5───X^0.5───
        """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            iy(q),
            iy(q)**-1,
            iy(q)**3,
            iy(q)**4.5,
            iy(q)**4.500001,
        ), """
q: ───Y───Y───Y───Y^0.5───Y^0.5───
    """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            iz(q),
            iz(q)**-1,
            iz(q)**3,
            iz(q)**4.5,
            iz(q)**4.500001,
        ), """
q: ───Z───Z───Z───S───S───
    """)


def test_rxyz_circuit_diagram():
    q = cirq.NamedQubit('q')

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.Rx(np.pi).on(q),
            cirq.Rx(-np.pi).on(q),
            cirq.Rx(-np.pi + 0.00001).on(q),
            cirq.Rx(-np.pi - 0.00001).on(q),
            cirq.Rx(3*np.pi).on(q),
            cirq.Rx(7*np.pi/2).on(q),
            cirq.Rx(9*np.pi/2 + 0.00001).on(q),
        ), """
q: ───Rx(π)───Rx(-π)───Rx(-π)───Rx(-π)───Rx(-π)───Rx(-0.5π)───Rx(0.5π)───
    """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.Rx(np.pi).on(q),
            cirq.Rx(np.pi/2).on(q),
            cirq.Rx(-np.pi + 0.00001).on(q),
            cirq.Rx(-np.pi - 0.00001).on(q),
        ), """
q: ---Rx(pi)---Rx(0.5pi)---Rx(-pi)---Rx(-pi)---
        """,
        use_unicode_characters=False)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.Ry(np.pi).on(q),
            cirq.Ry(-np.pi).on(q),
            cirq.Ry(3 * np.pi).on(q),
            cirq.Ry(9*np.pi/2).on(q),
        ), """
q: ───Ry(π)───Ry(-π)───Ry(-π)───Ry(0.5π)───
    """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.Rz(np.pi).on(q),
            cirq.Rz(-np.pi).on(q),
            cirq.Rz(3 * np.pi).on(q),
            cirq.Rz(9*np.pi/2).on(q),
            cirq.Rz(9*np.pi/2 + 0.00001).on(q),
        ), """
q: ───Rz(π)───Rz(-π)───Rz(-π)───Rz(0.5π)───Rz(0.5π)───
    """)


def test_trace_distance():
    foo = sympy.Symbol('foo')
    sx = cirq.X**foo
    sy = cirq.Y**foo
    sz = cirq.Z**foo
    sh = cirq.H**foo
    scx = cirq.CX**foo
    scz = cirq.CZ**foo
    sswap = cirq.SWAP**foo
    siswap = cirq.ISWAP**foo
    # These values should have 1.0 or 0.0 directly returned
    assert cirq.trace_distance_bound(sx) == 1.0
    assert cirq.trace_distance_bound(sy) == 1.0
    assert cirq.trace_distance_bound(sz) == 1.0
    assert cirq.trace_distance_bound(scx) == 1.0
    assert cirq.trace_distance_bound(scz) == 1.0
    assert cirq.trace_distance_bound(sswap) == 1.0
    assert cirq.trace_distance_bound(siswap) == 1.0
    assert cirq.trace_distance_bound(sh) == 1.0
    assert cirq.trace_distance_bound(cirq.I) == 0.0
    # These values are calculated, so we use approx_eq
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.X), 1.0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.Y**-1), 1.0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.Z**0.5),
                          np.sin(np.pi / 4))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.H**0.25),
                          np.sin(np.pi / 8))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CX**2), 0.0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CZ**(1 / 9)),
                          np.sin(np.pi / 18))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.SWAP**0.3),
                          np.sin(0.3 * np.pi / 2))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.ISWAP**0), 0.0)
