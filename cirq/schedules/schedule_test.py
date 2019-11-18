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

from datetime import timedelta
import pytest
import numpy as np

import cirq
from cirq.devices import UNCONSTRAINED_DEVICE


def test_equality():
    et = cirq.testing.EqualsTester()

    def simple_schedule(q, start_picos=0, duration_picos=1, num_ops=1):
        time_picos = start_picos
        scheduled_ops = []
        for _ in range(num_ops):
            op = cirq.ScheduledOperation(cirq.Timestamp(picos=time_picos),
                                         cirq.Duration(picos=duration_picos),
                                         cirq.H(q))
            scheduled_ops.append(op)
            time_picos += duration_picos
        return cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=scheduled_ops)

    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    et.make_equality_group(lambda: simple_schedule(q0))
    et.make_equality_group(lambda: simple_schedule(q1))
    et.make_equality_group(lambda: simple_schedule(q0, start_picos=1000))
    et.make_equality_group(lambda: simple_schedule(q0, duration_picos=1000))
    et.make_equality_group(lambda: simple_schedule(q0, num_ops=3))
    et.make_equality_group(lambda: simple_schedule(q1, num_ops=3))


def test_equality_timedelta():
    et = cirq.testing.EqualsTester()

    def simple_schedule(q, start_picos=0, duration_micros=1, num_ops=1):
        time_picos = start_picos
        scheduled_ops = []
        for _ in range(num_ops):
            op = cirq.ScheduledOperation(
                cirq.Timestamp(picos=time_picos),
                timedelta(microseconds=duration_micros), cirq.H(q))
            scheduled_ops.append(op)
            time_picos += duration_micros * 10**6
        return cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=scheduled_ops)

    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    et.make_equality_group(lambda: simple_schedule(q0))
    et.make_equality_group(lambda: simple_schedule(q1))
    et.make_equality_group(lambda: simple_schedule(q0, start_picos=1000))
    et.make_equality_group(lambda: simple_schedule(q0, duration_micros=5))
    et.make_equality_group(lambda: simple_schedule(q0, num_ops=3))
    et.make_equality_group(lambda: simple_schedule(q1, num_ops=3))


def test_query_point_operation_inclusive():
    q = cirq.NamedQubit('q')
    zero = cirq.Timestamp(picos=0)
    ps = cirq.Duration(picos=1)
    op = cirq.ScheduledOperation(zero, cirq.Duration(), cirq.H(q))
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=[op])

    def query(t, d=cirq.Duration(), qubits=None):
        return schedule.query(time=t,
                              duration=d,
                              qubits=qubits,
                              include_query_end_time=True,
                              include_op_end_times=True)

    assert query(zero) == [op]
    assert query(zero + ps) == []
    assert query(zero - ps) == []

    assert query(zero, qubits=[]) == []
    assert query(zero, qubits=[q]) == [op]

    assert query(zero, ps) == [op]
    assert query(zero - 0.5 * ps, ps) == [op]
    assert query(zero - ps, ps) == [op]
    assert query(zero - 2 * ps, ps) == []
    assert query(zero - ps, 3 * ps) == [op]
    assert query(zero + ps, ps) == []


def test_query_point_operation_exclusive():
    q = cirq.NamedQubit('q')
    zero = cirq.Timestamp(picos=0)
    ps = cirq.Duration(picos=1)
    op = cirq.ScheduledOperation(zero, cirq.Duration(), cirq.H(q))
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=[op])

    assert schedule.query(time=zero,
                          include_query_end_time=False,
                          include_op_end_times=False) == []
    assert schedule.query(time=zero + ps,
                          include_query_end_time=False,
                          include_op_end_times=False) == []
    assert schedule.query(time=zero - ps,
                          include_query_end_time=False,
                          include_op_end_times=False) == []
    assert schedule.query(time=zero) == []
    assert schedule.query(time=zero + ps) == []
    assert schedule.query(time=zero - ps) == []

    assert schedule.query(time=zero, qubits=[]) == []
    assert schedule.query(time=zero, qubits=[q]) == []

    assert schedule.query(time=zero, duration=ps) == []
    assert schedule.query(time=zero - 0.5 * ps, duration=ps) == [op]
    assert schedule.query(time=zero - ps, duration=ps) == []
    assert schedule.query(time=zero - 2 * ps, duration=ps) == []
    assert schedule.query(time=zero - ps, duration=3 * ps) == [op]
    assert schedule.query(time=zero + ps, duration=ps) == []


def test_query_overlapping_operations_inclusive():
    q = cirq.NamedQubit('q')
    zero = cirq.Timestamp(picos=0)
    ps = cirq.Duration(picos=1)
    op1 = cirq.ScheduledOperation(zero, 2 * ps, cirq.H(q))
    op2 = cirq.ScheduledOperation(zero + ps, 2 * ps, cirq.H(q))
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=[op2, op1])

    def query(t, d=cirq.Duration(), qubits=None):
        return schedule.query(time=t,
                              duration=d,
                              qubits=qubits,
                              include_query_end_time=True,
                              include_op_end_times=True)

    assert query(zero - 0.5 * ps, ps) == [op1]
    assert query(zero - 0.5 * ps, 2 * ps) == [op1, op2]
    assert query(zero, ps) == [op1, op2]
    assert query(zero + 0.5 * ps, ps) == [op1, op2]
    assert query(zero + ps, ps) == [op1, op2]
    assert query(zero + 1.5 * ps, ps) == [op1, op2]
    assert query(zero + 2.0 * ps, ps) == [op1, op2]
    assert query(zero + 2.5 * ps, ps) == [op2]
    assert query(zero + 3.0 * ps, ps) == [op2]
    assert query(zero + 3.5 * ps, ps) == []


def test_query_overlapping_operations_exclusive():
    q = cirq.NamedQubit('q')
    zero = cirq.Timestamp(picos=0)
    ps = cirq.Duration(picos=1)
    op1 = cirq.ScheduledOperation(zero, 2 * ps, cirq.H(q))
    op2 = cirq.ScheduledOperation(zero + ps, 2 * ps, cirq.H(q))
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=[op2, op1])

    assert schedule.query(time=zero - 0.5 * ps, duration=ps) == [op1]
    assert schedule.query(time=zero - 0.5 * ps, duration=2 * ps) == [op1, op2]
    assert schedule.query(time=zero, duration=ps) == [op1]
    assert schedule.query(time=zero + 0.5 * ps, duration=ps) == [op1, op2]
    assert schedule.query(time=zero + ps, duration=ps) == [op1, op2]
    assert schedule.query(time=zero + 1.5 * ps, duration=ps) == [op1, op2]
    assert schedule.query(time=zero + 2.0 * ps, duration=ps) == [op2]
    assert schedule.query(time=zero + 2.5 * ps, duration=ps) == [op2]
    assert schedule.query(time=zero + 3.0 * ps, duration=ps) == []
    assert schedule.query(time=zero + 3.5 * ps, duration=ps) == []


def test_query_timedelta():
    q = cirq.NamedQubit('q')
    zero = cirq.Timestamp(picos=0)
    ms = timedelta(microseconds=1000)
    op1 = cirq.ScheduledOperation(zero, 2 * ms, cirq.H(q))
    op2 = cirq.ScheduledOperation(zero + ms, 2 * ms, cirq.H(q))
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=[op2, op1])

    def query(t, d=timedelta(), qubits=None):
        return schedule.query(time=t,
                              duration=d,
                              qubits=qubits,
                              include_query_end_time=True,
                              include_op_end_times=True)

    assert query(zero - 0.5 * ms, ms) == [op1]
    assert query(zero - 0.5 * ms, 2 * ms) == [op1, op2]
    assert query(zero, ms) == [op1, op2]
    assert query(zero + 0.5 * ms, ms) == [op1, op2]
    assert query(zero + ms, ms) == [op1, op2]
    assert query(zero + 1.5 * ms, ms) == [op1, op2]
    assert query(zero + 2.0 * ms, ms) == [op1, op2]
    assert query(zero + 2.5 * ms, ms) == [op2]
    assert query(zero + 3.0 * ms, ms) == [op2]
    assert query(zero + 3.5 * ms, ms) == []


def test_slice_operations():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    zero = cirq.Timestamp(picos=0)
    ps = cirq.Duration(picos=1)
    op1 = cirq.ScheduledOperation(zero, ps, cirq.H(q0))
    op2 = cirq.ScheduledOperation(zero + 2 * ps, 2 * ps, cirq.CZ(q0, q1))
    op3 = cirq.ScheduledOperation(zero + 10 * ps, ps, cirq.H(q1))
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=[op1, op2, op3])

    assert schedule[zero] == [op1]
    assert schedule[zero + ps*0.5] == [op1]
    assert schedule[zero:zero] == []
    assert schedule[zero + ps*0.5:zero + ps*0.5] == [op1]
    assert schedule[zero:zero + ps] == [op1]
    assert schedule[zero:zero + 2 * ps] == [op1]
    assert schedule[zero:zero + 2.1 * ps] == [op1, op2]
    assert schedule[zero:zero + 20 * ps] == [op1, op2, op3]
    assert schedule[zero + 2.5 * ps:zero + 20 * ps] == [op2, op3]
    assert schedule[zero + 5 * ps:zero + 20 * ps] == [op3]


def test_include():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    zero = cirq.Timestamp(picos=0)
    ps = cirq.Duration(picos=1)
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE)

    op0 = cirq.ScheduledOperation(zero, ps, cirq.H(q0))
    schedule.include(op0)
    with pytest.raises(ValueError):
        schedule.include(cirq.ScheduledOperation(zero, ps, cirq.H(q0)))
    with pytest.raises(ValueError):
        schedule.include(cirq.ScheduledOperation(zero + 0.5 * ps,
                                                 ps,
                                                 cirq.H(q0)))
    op1 = cirq.ScheduledOperation(zero + 2 * ps, ps, cirq.H(q0))
    schedule.include(op1)
    op2 = cirq.ScheduledOperation(zero + 0.5 * ps, ps, cirq.H(q1))
    schedule.include(op2)

    assert schedule.query(time=zero, duration=ps * 10) == [op0, op2, op1]


def test_exclude():
    q = cirq.NamedQubit('q')
    zero = cirq.Timestamp(picos=0)
    ps = cirq.Duration(picos=1)
    op = cirq.ScheduledOperation(zero, ps, cirq.H(q))
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=[op])

    assert not schedule.exclude(cirq.ScheduledOperation(zero + ps,
                                                        ps,
                                                        cirq.H(q)))
    assert not schedule.exclude(cirq.ScheduledOperation(zero, ps, cirq.X(q)))
    assert schedule.query(time=zero, duration=ps * 10) == [op]
    assert schedule.exclude(cirq.ScheduledOperation(zero, ps, cirq.H(q)))
    assert schedule.query(time=zero, duration=ps * 10) == []
    assert not schedule.exclude(cirq.ScheduledOperation(zero, ps, cirq.H(q)))


def test_unitary():
    q = cirq.NamedQubit('q')
    zero = cirq.Timestamp(picos=0)
    ps = cirq.Duration(picos=1)
    op = cirq.ScheduledOperation(zero, ps, cirq.H(q))
    schedule = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                             scheduled_operations=[op])

    cirq.testing.assert_has_consistent_apply_unitary(schedule)
    np.testing.assert_allclose(cirq.unitary(schedule), cirq.unitary(cirq.H))
    assert cirq.has_unitary(schedule)

    schedule2 = cirq.Schedule(device=UNCONSTRAINED_DEVICE,
                              scheduled_operations=[
                                  cirq.ScheduledOperation(
                                      zero, ps,
                                      cirq.depolarize(0.5).on(q))
                              ])
    assert not cirq.has_unitary(schedule2)
