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

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    TypeVar,
    Optional,
    Tuple,
    List,
    Sequence,
)

from collections import defaultdict
import numpy as np
from typing_extensions import Protocol

from cirq.protocols import qid_shape_protocol

if TYPE_CHECKING:
    import cirq

TDefault = TypeVar('TDefault')


class SupportsExplicitHasUnitary(Protocol):
    """An object that explicitly specifies whether it has a unitary effect."""

    def _has_unitary_(self) -> bool:
        """Determines whether the receiver has a unitary effect.

        This method is used preferentially by the global `cirq.has_unitary`
        method, because this method is much cheaper than the fallback strategies
        such as checking `value._unitary_` (which causes a large matrix to be
        computed).

        Returns:
            Whether or not the receiving object (`self`) has a unitary effect.
        """


def has_unitary(val: Any) -> bool:
    """Determines whether the value has a unitary effect.

    Determines whether `val` has a unitary effect by attempting the following
    strategies:

    1. Try to use `val.has_unitary()`.
        Case a) Method not present or returns `NotImplemented`.
            Inconclusive.
        Case b) Method returns `True`.
            Unitary.
        Case c) Method returns `False`.
            Not unitary.

    2. Try to use `val._decompose_()`.
        Case a) Method not present or returns `NotImplemented` or `None`.
            Inconclusive.
        Case b) Method returns an OP_TREE containing only unitary operations.
            Unitary.
        Case c) Method returns an OP_TREE containing non-unitary operations.
            Not Unitary.

    3. Try to use `val._apply_unitary_(args)`.
        Case a) Method not present or returns `NotImplemented`.
            Inconclusive.
        Case b) Method returns a numpy array.
            Unitary.
        Case c) Method returns `None`.
            Not unitary.

    4. Try to use `val._unitary_()`.
        Case a) Method not present or returns `NotImplemented`.
            Continue to next strategy.
        Case b) Method returns a numpy array.
            Unitary.
        Case c) Method returns `None`.
            Not unitary.

    It is assumed that, when multiple of these strategies give a conclusive
    result, that these results will all be consistent with each other. If all
    strategies are inconclusive, the value is classified as non-unitary.

    Args:
        The value that may or may not have a unitary effect.

    Returns:
        Whether or not `val` has a unitary effect.
    """
    strats = [
        _strat_has_unitary_from_has_unitary, _strat_has_unitary_from_decompose,
        _strat_has_unitary_from_apply_unitary, _strat_has_unitary_from_unitary
    ]
    for strat in strats:
        result = strat(val)
        if result is not None:
            return result

    # If you can't tell that it's unitary, it's not unitary.
    return False


def _strat_has_unitary_from_has_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _has_unitary_ method."""
    if hasattr(val, '_has_unitary_'):
        result = val._has_unitary_()
        if result is NotImplemented:
            return None
        return result
    return None


def _strat_has_unitary_from_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _unitary_ method."""
    getter = getattr(val, '_unitary_', None)
    if getter is None:
        return None
    result = getter()
    return result is not NotImplemented and result is not None


def _strat_has_unitary_from_decompose(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _decompose_ method."""
    operations, _, _ = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return None
    return all(has_unitary(op) for op in operations)


def _strat_has_unitary_from_apply_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _apply_unitary_ method.
    """
    from cirq.protocols.apply_unitary import ApplyUnitaryArgs
    from cirq import linalg

    method = getattr(val, '_apply_unitary_', None)
    if method is None:
        return None

    val_qid_shape = qid_shape_protocol.qid_shape(val, None)
    if val_qid_shape is None:
        return None
    state = linalg.one_hot(shape=val_qid_shape, dtype=np.complex64)
    buffer = np.empty_like(state)
    result = method(ApplyUnitaryArgs(state, buffer, range(len(val_qid_shape))))
    if result is NotImplemented:
        return None
    return result is not None


def _try_decompose_into_operations_and_qubits(val: Any) -> Tuple[Optional[
        List['cirq.Operation']], Sequence['cirq.Qid'], Tuple[int, ...]]:
    """Returns the value's decomposition (if any) and the qubits it applies to.
    """
    from cirq.protocols.decompose import (decompose_once,
                                          decompose_once_with_qubits)
    from cirq import LineQid, Gate, Operation

    if isinstance(val, Gate):
        # Gates don't specify qubits, and so must be handled specially.
        qid_shape = qid_shape_protocol.qid_shape(val)
        qubits = LineQid.for_qid_shape(qid_shape)  # type: Sequence[cirq.Qid]
        return decompose_once_with_qubits(val, qubits, None), qubits, qid_shape

    if isinstance(val, Operation):
        qid_shape = qid_shape_protocol.qid_shape(val)
        return decompose_once(val, None), val.qubits, qid_shape

    result = decompose_once(val, None)
    if result is not None:
        qubit_set = set()
        qid_shape_dict = defaultdict(lambda: 1)  # type: Dict[cirq.Qid, int]
        for op in result:
            for level, q in zip(qid_shape_protocol.qid_shape(op), op.qubits):
                qubit_set.add(q)
                qid_shape_dict[q] = max(qid_shape_dict[q], level)
        qubits = sorted(qubit_set)
        return result, qubits, tuple(qid_shape_dict[q] for q in qubits)

    return None, (), ()
