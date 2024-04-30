from functools import partial


class TwoLevelMergeIndex:
    def __init__(self, outer_cls, inner_cls, intersect_strategy='ReturnEntry'):
        self._outer = outer_cls()
        self._inner_cls = inner_cls
        self.where_intersect = partial(strategies[intersect_strategy], self._outer)

    def add(self, key, entry) -> None:
        if (inner_idx := self._outer.where_contain(key[0])) is None:
            inner_idx = self._inner_cls()
            self._outer.add(key[0], inner_idx)
        inner_idx.add(key[1], entry)

    def where_contain(self, key):
        return (inner := self._outer.where_contain(key[0])) and inner.where_contain(key[1])


def intersect_return_entry(outer_idx, bboxes):
    """
    :param outer_idx: the top level index.
    :param bboxes: (dim0_min, dim0_max, dim1_min, dim1_max, ...).
    :return: a generator that simply searches respecting $bboxes
    """
    for inner_idx in outer_idx.where_intersect(bboxes[0]):
        yield from inner_idx.where_intersect(bboxes[1])


def intersect_fuzzy_search(outer_idx, bboxes):
    """
    linear constrains search.
    :param outer_idx: the top level index
    :param bboxes: just care (dim2_min, dim2_max).
    :return: a generator that searches all possible region.
    """
    lo, hi = bboxes
    for outer_key, inner_idx in outer_idx:
        yield from inner_idx.where_intersect((lo - outer_key + 1, hi))


def intersect_not_sure_search(outer_idx, bboxes):
    """
    separate absolute inner entries from the unsure.
    :param outer_idx: the top level index.
    :param bboxes: (dim0_min, dim0_max, dim1_min, dim1_max, ...).
    :return: a tuple of generators: (not_sure, sure)
    """
    candidates, probation = outer_idx.where_intersect(bboxes[0])
    return (entry for inner in candidates for entry in inner.where_intersect(bboxes[1])), \
        (entry for inner in probation for entry in inner.where_intersect(bboxes[1]))


def intersect_fuzzy_inner_all(outer_idx, bboxes):
    """
    :param outer_idx: the top level index.
    :param bboxes: (min, max).
    :return: a generator that iterates inner search result as a whole set with the linear constraints.
    """
    lo, hi = bboxes
    for outer_key, inner_idx in outer_idx:
        yield inner_idx.where_intersect((lo - outer_key + 1, hi))


def intersect_not_sure_group(outer_idx, bboxes):
    """
    separate absolute inner entries from the unsure. results are group by regions
    :param outer_idx: the top level index.
    :param bboxes: (dim0_min, dim0_max, dim1_min, dim1_max, ...).
    :return: a tuple: (not_sure, sure), where $not_sure and $sure are also tuples
        with the same format: (region, entries), with $entries a generator
    """
    candidates, probation = outer_idx.where_intersect(bboxes[0])
    return ((inner, inner.where_intersect(bboxes[1])) for inner in candidates), \
        ((inner, inner.where_intersect(bboxes[1])) for inner in probation)


strategies = {name.removeprefix('intersect_').title().replace('_', ''): stra for name, stra in globals().items()
              if name.startswith('intersect_')}
