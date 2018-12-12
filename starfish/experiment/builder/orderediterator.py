from itertools import product
from typing import Iterator, Mapping, Sequence, Tuple

from starfish.types import Indices


def join_dimension_ids(
        dimension_order: Sequence[Indices],
        *,
        rounds: Sequence[int],
        chs: Sequence[int],
        zlayers: Sequence[int],
) -> Sequence[Tuple[Indices, Sequence[int]]]:
    """
    Given a sequence of dimensions and their ids, return a sequence of tuples of dimensions and
    its respective ids.

    For example, if dimension_sequence is (ROUND, CH, Z) and each dimension has ids [0, 1], return
    ((ROUND, [0, 1]), (CH, [0, 1]), (Z, [0, 1]).
    """
    dimension_mapping = {
        Indices.ROUND: rounds,
        Indices.CH: chs,
        Indices.Z: zlayers,
    }

    return [
        (dimension, dimension_mapping[dimension])
        for dimension in dimension_order
    ]


def ordered_iterator(
        dimension_ids: Sequence[Tuple[Indices, Sequence[int]]]
) -> Iterator[Mapping[Indices, int]]:
    """
    Given a sequence of tuples of dimensions and its respective sequence of ids, return an iterator
    that steps through all the possible points in the space.  The sequence is ordered from the
    slowest varying dimension to the fastest varying dimension.
    """
    for tpl in product(*[dimension_ids for _, dimension_ids in dimension_ids]):
        yield dict(zip((index for index, _ in dimension_ids), tpl))
