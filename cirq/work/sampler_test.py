# Copyright 2019 The Cirq Developers
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
"""Tests for cirq.Sampler."""
import sympy

import cirq
from cirq.work.sampler import _trial_results_to_dataframe


def test_sampler_fail():

    class FailingSampler(cirq.Sampler):

        def run_sweep(self, program, params, repetitions: int = 1):
            raise ValueError('test')

    cirq.testing.assert_asyncio_will_raise(FailingSampler().run_async(
        cirq.Circuit(), repetitions=1),
                                           ValueError,
                                           match='test')


def test_frame_trial_results():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    q = cirq.NamedQubit('q')
    p = cirq.NamedQubit('p')
    c = cirq.Circuit.from_ops(
        cirq.X(q)**a,
        cirq.CNOT(q, p),
        cirq.X(p)**b,
        cirq.measure(q, key='m'),
        cirq.measure(p, key='m2'),
    )

    results = cirq.Simulator().run_sweep(c, repetitions=5, params=cirq.Linspace('a', 0, 3, 5) * cirq.Linspace('b', -1, +1, 5))
    x = _trial_results_to_dataframe(results)
    print(x)
    print(x[x['a'] == 1])
    assert False
