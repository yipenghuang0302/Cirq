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

"""Basic types defining qubits, gates, and operations."""

from typing import (Any, Callable, Collection, Optional, Sequence, Tuple,
                    TYPE_CHECKING, Union)

import abc
import functools

from cirq import value, protocols

if TYPE_CHECKING:
    from cirq.ops import gate_operation, linear_combinations


class Qid(metaclass=abc.ABCMeta):
    """Identifies a quantum object such as a qubit, qudit, resonator, etc.

    Child classes represent specific types of objects, such as a qubit at a
    particular location on a chip or a qubit with a particular name.

    The main criteria that a custom qid must satisfy is *comparability*. Child
    classes meet this criteria by implementing the `_comparison_key` method. For
    example, `cirq.LineQubit`'s `_comparison_key` method returns `self.x`. This
    ensures that line qubits with the same `x` are equal, and that line qubits
    will be sorted ascending by `x`. `Qid` implements all equality,
    comparison, and hashing methods via `_comparison_key`.
    """

    @abc.abstractmethod
    def _comparison_key(self) -> Any:
        """Returns a value used to sort and compare this qubit with others.

        By default, qubits of differing type are sorted ascending according to
        their type name. Qubits of the same type are then sorted using their
        comparison key.
        """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the dimension or the number of quantum levels this qid has.
        E.g. 2 for a qubit, 3 for a qutrit, etc.
        """

    @staticmethod
    def validate_dimension(dimension: int) -> None:
        """Raises an exception if `dimension` is not positive.

        Raises:
            ValueError: `dimension` is not positive.
        """
        if dimension < 1:
            raise ValueError(
                'Wrong qid dimension. '
                'Expected a positive integer but got {}.'.format(dimension))

    def with_dimension(self, dimension: int) -> 'Qid':
        """Returns a new qid with a different dimension.

        Child classes can override.  Wraps the qubit object by default.

        Args:
            dimension: The new dimension or number of levels.
        """
        if dimension == self.dimension:
            return self
        return _QubitAsQid(self, dimension=dimension)

    def _cmp_tuple(self):
        return (type(self).__name__, repr(type(self)), self._comparison_key(),
                self.dimension)

    def __hash__(self):
        return hash((Qid, self._comparison_key()))

    def __eq__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() == other._cmp_tuple()

    def __ne__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() != other._cmp_tuple()

    def __lt__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() < other._cmp_tuple()

    def __gt__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() > other._cmp_tuple()

    def __le__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() <= other._cmp_tuple()

    def __ge__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() >= other._cmp_tuple()


@functools.total_ordering
class _QubitAsQid(Qid):

    def __init__(self, qubit: Qid, dimension: int):
        self._qubit = qubit
        self._dimension = dimension
        self.validate_dimension(dimension)

    @property
    def qubit(self) -> Qid:
        return self._qubit

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> Qid:
        """Returns a copy with a different dimension or number of levels."""
        return self.qubit.with_dimension(dimension)

    def _comparison_key(self) -> Any:
        # Don't include self._qubit.dimension
        return self._qubit._cmp_tuple()[:-1]

    def __repr__(self):
        return '{!r}.with_dimension({})'.format(self.qubit, self.dimension)

    def __str__(self):
        return '{!s} (d={})'.format(self.qubit, self.dimension)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['qubit', 'dimension'])


class Gate(metaclass=value.ABCMetaImplementAnyOneOf):
    """An operation type that can be applied to a collection of qubits.

    Gates can be applied to qubits by calling their on() method with
    the qubits to be applied to supplied, or, alternatively, by simply
    calling the gate on the qubits.  In other words calling MyGate.on(q1, q2)
    to create an Operation on q1 and q2 is equivalent to MyGate(q1,q2).

    Gates operate on a certain number of qubits. All implementations of gate
    must implement the `num_qubits` method declaring how many qubits they
    act on. The gate feature classes `SingleQubitGate` and `TwoQubitGate`
    can be used to avoid writing this boilerplate.

    Linear combinations of gates can be created by adding gates together and
    multiplying them by scalars.
    """

    def validate_args(self, qubits: Sequence[Qid]) -> None:
        """Checks if this gate can be applied to the given qubits.

        By default checks that:
        - inputs are of type `Qid`
        - len(qubits) == num_qubits()
        - qubit_i.dimension == qid_shape[i] for all qubits

        Child classes can override.  The child implementation should call
        `super().validate_args(qubits)` then do custom checks.

        Args:
            qubits: The sequence of qubits to potentially apply the gate to.

        Throws:
            ValueError: The gate can't be applied to the qubits.
        """
        _validate_qid_shape(self, qubits)

    def on(self, *qubits: Qid) -> 'Operation':
        """Returns an application of this gate to the given qubits.

        Args:
            *qubits: The collection of qubits to potentially apply the gate to.
        """
        # Avoids circular import.
        from cirq.ops import gate_operation
        return gate_operation.GateOperation(self, list(qubits))

    def wrap_in_linear_combination(
            self,
            coefficient: Union[complex, float, int]=1
            ) -> 'linear_combinations.LinearCombinationOfGates':
        from cirq.ops import linear_combinations
        return linear_combinations.LinearCombinationOfGates({self: coefficient})

    def __add__(self,
                other: Union['Gate',
                             'linear_combinations.LinearCombinationOfGates']
                ) -> 'linear_combinations.LinearCombinationOfGates':
        if isinstance(other, Gate):
            return (self.wrap_in_linear_combination() +
                    other.wrap_in_linear_combination())
        return self.wrap_in_linear_combination() + other

    def __sub__(self,
                other: Union['Gate',
                             'linear_combinations.LinearCombinationOfGates']
                ) -> 'linear_combinations.LinearCombinationOfGates':
        if isinstance(other, Gate):
            return (self.wrap_in_linear_combination() -
                    other.wrap_in_linear_combination())
        return self.wrap_in_linear_combination() - other

    def __neg__(self) -> 'linear_combinations.LinearCombinationOfGates':
        return self.wrap_in_linear_combination(coefficient=-1)

    def __mul__(self, other: Union[complex, float, int]
                ) -> 'linear_combinations.LinearCombinationOfGates':
        return self.wrap_in_linear_combination(coefficient=other)

    def __rmul__(self, other: Union[complex, float, int]
                 ) -> 'linear_combinations.LinearCombinationOfGates':
        return self.wrap_in_linear_combination(coefficient=other)

    def __truediv__(self, other: Union[complex, float, int]
                    ) -> 'linear_combinations.LinearCombinationOfGates':
        return self.wrap_in_linear_combination(coefficient=1 / other)

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            # HACK: break cycle
            from cirq.devices import line_qubit

            decomposed = protocols.decompose_once_with_qubits(
                self, qubits=line_qubit.LineQid.for_gate(self), default=None)
            if decomposed is None:
                return NotImplemented

            inverse_decomposed = protocols.inverse(decomposed, None)
            if inverse_decomposed is None:
                return NotImplemented

            return _InverseCompositeGate(self)

        return NotImplemented

    def __call__(self, *args, **kwargs):
        return self.on(*args, **kwargs)

    def controlled_by(self,
                      *control_qubits: Qid,
                      control_values: Optional[Sequence[
                          Union[int, Collection[int]]]] = None) -> 'Gate':
        """Returns a controlled version of this gate.

        Args:
            control_qubits: Optional qubits to control the gate by.
        """
        # Avoids circular import.
        from cirq.ops import ControlledGate
        if len(control_qubits) == 0:
            return self
        return ControlledGate(self, control_qubits, len(control_qubits),
                              control_values)

    # num_qubits, _num_qubits_, and _qid_shape_ are implemented with alternative
    # to keep backwards compatibility with versions of cirq where num_qubits
    # is an abstract method.
    def _backwards_compatibility_num_qubits(self) -> int:
        return protocols.num_qubits(self)

    @value.alternative(requires='_num_qubits_',
                       implementation=_backwards_compatibility_num_qubits)
    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""

    def _num_qubits_from_shape(self) -> int:
        shape = self._qid_shape_()
        if shape is NotImplemented:
            return NotImplemented
        return len(shape)

    def _num_qubits_proto_from_num_qubits(self) -> int:
        return self.num_qubits()

    @value.alternative(requires='num_qubits',
                       implementation=_num_qubits_proto_from_num_qubits)
    @value.alternative(requires='_qid_shape_',
                       implementation=_num_qubits_from_shape)
    def _num_qubits_(self) -> int:
        """The number of qubits this gate acts on."""

    def _default_shape_from_num_qubits(self) -> Tuple[int, ...]:
        num_qubits = self._num_qubits_()
        if num_qubits is NotImplemented:
            return NotImplemented
        return (2,) * num_qubits

    @value.alternative(requires='_num_qubits_',
                       implementation=_default_shape_from_num_qubits)
    def _qid_shape_(self) -> Tuple[int, ...]:
        """Returns a Tuple containing the number of quantum levels of each qid
        the gate acts on.  E.g. (2, 2, 2) for the three-qubit CCZ gate and
        (3, 3) for a 2-qutrit ternary gate.
        """

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, attribute_names=[])


class Operation(metaclass=abc.ABCMeta):
    """An effect applied to a collection of qubits.

    The most common kind of Operation is a GateOperation, which separates its
    effect into a qubit-independent Gate and the qubits it should be applied to.
    """

    @property
    @abc.abstractmethod
    def qubits(self) -> Tuple[Qid, ...]:
        raise NotImplementedError()

    def _num_qubits_(self) -> int:
        """The number of qubits this operation acts on.

        By definition, returns the length of `qubits`.
        """
        return len(self.qubits)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return protocols.qid_shape(self.qubits)

    @abc.abstractmethod
    def with_qubits(self, *new_qubits: Qid) -> 'Operation':
        pass

    def transform_qubits(self, func: Callable[[Qid], Qid]) -> 'Operation':
        """Returns the same operation, but with different qubits.

        Args:
            func: The function to use to turn each current qubit into a desired
                new qubit.

        Returns:
            The receiving operation but with qubits transformed by the given
                function.
        """
        return self.with_qubits(*(func(q) for q in self.qubits))

    def controlled_by(self,
                      *control_qubits: Qid,
                      control_values: Optional[Sequence[
                          Union[int, Collection[int]]]] = None) -> 'Operation':
        """Returns a controlled version of this operation.

        Args:
            control_qubits: Qubits to control the operation by. Required.
        """
        # Avoids circular import.
        from cirq.ops import ControlledOperation
        if len(control_qubits) == 0:
            return self
        return ControlledOperation(control_qubits, self, control_values)

    def validate_args(self, qubits: Sequence[Qid]):
        """Raises an exception if the `qubits` don't match this operation's qid
        shape.

        Call this method from a subclass's `with_qubits` method.

        Args:
            qubits: The new qids for the operation.

        Raises:
            ValueError: The operation had qids that don't match it's qid shape.
        """
        _validate_qid_shape(self, qubits)


@value.value_equality
class _InverseCompositeGate(Gate):
    """The inverse of a composite gate."""

    def __init__(self, original: Gate) -> None:
        self._original = original

    def _num_qubits_(self):
        return protocols.num_qubits(self._original)

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self._original
        return NotImplemented

    def _decompose_(self, qubits):
        return protocols.inverse(
            protocols.decompose_once_with_qubits(self._original, qubits))

    def _value_equality_values_(self):
        return self._original

    def __repr__(self):
        return '({!r}**-1)'.format(self._original)


def _validate_qid_shape(val: Any, qubits: Sequence[Qid]) -> None:
    """Helper function to validate qubits for gates and operations.

    Raises:
        ValueError: The operation had qids that don't match it's qid shape.
    """
    qid_shape = protocols.qid_shape(val)
    if len(qubits) != len(qid_shape):
        raise ValueError('Wrong number of qubits for <{!r}>. '
                         'Expected {} qubits but got <{!r}>.'.format(
                             val, len(qid_shape), qubits))
    if any(qid.dimension != dimension
           for qid, dimension in zip(qubits, qid_shape)):
        raise ValueError('Wrong shape of qids for <{!r}>. '
                         'Expected {} but got {} <{!r}>.'.format(
                             val, qid_shape,
                             tuple(qid.dimension for qid in qubits), qubits))
    if len(set(qubits)) != len(qubits):
        raise ValueError('Duplicate qids for <{!r}>. '
                         'Expected unique qids but got <{!r}>.'.format(
                             val, qubits))
