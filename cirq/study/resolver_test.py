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

"""Tests for parameter resolvers."""

import numpy as np
import sympy

import cirq


def test_value_of():
    assert not bool(cirq.ParamResolver())

    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1, 'c': 1 + 1j})
    assert bool(r)

    assert r.value_of('x') == sympy.Symbol('x')
    assert r.value_of('a') == 0.5
    assert r.value_of(sympy.Symbol('a')) == 0.5
    assert r.value_of(0.5) == 0.5
    assert r.value_of(sympy.Symbol('b')) == 0.1
    assert r.value_of(0.3) == 0.3
    assert r.value_of(sympy.Symbol('a') * 3) == 1.5
    assert r.value_of(sympy.Symbol('b') / 0.1 - sympy.Symbol('a')) == 0.5

    assert r.value_of(sympy.pi) == np.pi
    assert r.value_of(2 * sympy.pi) == 2 * np.pi
    assert r.value_of('c') == 1 + 1j
    assert r.value_of(sympy.I * sympy.pi) == np.pi * 1j


def test_param_dict():
    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1})
    r2 = cirq.ParamResolver(r)
    assert r2 is r
    assert r.param_dict == {'a': 0.5, 'b': 0.1}


def test_param_dict_iter():
    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1})
    assert [key for key in r] == ['a', 'b']
    assert [r.value_of(key) for key in r] == [0.5, 0.1]
    assert list(r) == ['a', 'b']


def test_equals():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(cirq.ParamResolver(),
                          cirq.ParamResolver(None),
                          cirq.ParamResolver({}),
                          cirq.ParamResolver(cirq.ParamResolver({})))
    et.make_equality_group(lambda: cirq.ParamResolver({'a': 0.0}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.0, 'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.3, 'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'c': 0.1}))


def test_repr():
    cirq.testing.assert_equivalent_repr(cirq.ParamResolver())
    cirq.testing.assert_equivalent_repr(cirq.ParamResolver({'a': 2.0}))
