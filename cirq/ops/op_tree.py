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

"""A recursive type describing trees of operations, and utility methods for it.
"""

from typing import Any, Callable, Iterable, Union

from cirq.ops.moment import Moment
from cirq.ops.raw_types import Operation


OP_TREE = Union[Operation, Iterable[Any]]
"""The recursive type consumed by circuit builder methods.

An OP_TREE is a contract, not a class. The basic idea is that, if the input can
be iteratively flattened into a list of operations, then the input is an
OP_TREE.

For example:
- An Operation is an OP_TREE all by itself.
- A list of operations is an OP_TREE.
- A list of tuples of operations is an OP_TREE.
- A list with a mix of operations and lists of operations is an OP_TREE.
- A generator yielding operations is an OP_TREE.

Note: once mypy has support for recursive types we can define this as:

OP_TREE = Union[Operation, Iterable['OP_TREE']]

See: https://github.com/python/mypy/issues/731
"""


def flatten_op_tree(root: OP_TREE,
                    preserve_moments: bool = False
                    ) -> Iterable[Union[Operation, Moment]]:
    """Performs an in-order iteration of the operations (leaves) in an OP_TREE.

    Args:
        root: The operation or tree of operations to iterate.
        preserve_moments: Whether to yield Moments intact instead of
            flattening them

    Yields:
        Operations from the tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if (isinstance(root, Operation)
            or preserve_moments and isinstance(root, Moment)):
        yield root
        return

    if isinstance(root, Iterable):
        for subtree in root:
            for item in flatten_op_tree(subtree, preserve_moments):
                yield item
        return

    raise TypeError('Not an Iterable or an Operation: {} {}'.format(
        type(root), root))


def transform_op_tree(
        root: OP_TREE,
        op_transformation: Callable[[Operation], OP_TREE]=lambda e: e,
        iter_transformation: Callable[[Iterable[OP_TREE]], OP_TREE]=lambda e: e,
        preserve_moments: bool = False
) -> OP_TREE:
    """Maps transformation functions onto the nodes of an OP_TREE.

    Args:
        root: The operation or tree of operations to transform.
        op_transformation: How to transform the operations (i.e. leaves).
        iter_transformation: How to transform the iterables (i.e. internal
            nodes).
        preserve_moments: Whether to leave Moments alone. If True, the
            transformation functions will not be applied to Moments or the
            operations within them.

    Returns:
        A transformed operation tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if isinstance(root, Operation):
        return op_transformation(root)

    if preserve_moments and isinstance(root, Moment):
        return root

    if isinstance(root, Iterable):
        return iter_transformation(
            transform_op_tree(subtree,
                              op_transformation,
                              iter_transformation,
                              preserve_moments)
            for subtree in root)

    raise TypeError('Not an Iterable or an Operation: {}'.format(root))


def freeze_op_tree(root: OP_TREE) -> OP_TREE:
    """Replaces all iterables in the OP_TREE with tuples.

    Args:
        root: The operation or tree of operations to freeze.

    Returns:
        An OP_TREE with the same operations and branching structure, but where
        all internal nodes are tuples instead of arbitrary iterables.
    """
    return transform_op_tree(root, iter_transformation=tuple)
