from typing import NamedTuple

from search.baseline.state import state_maintain


class Mock(NamedTuple):
    start: int
    end: int


def test_state_maintain():
    intervals = [[9, 10], [8, 9], [7, 7], [3, 5]]
    mocks = [Mock(*inter) for inter in intervals]
    cur = Mock(11, 11)
    state_maintainer = state_maintain(3)
    fruits, remain = state_maintainer(11, mocks, [cur, cur], 0, False)
    assert list(fruits) == mocks[1:] and list(remain) == [] and state_maintainer.__closure__[0].cell_contents == 2
    fruits, remain = state_maintainer(11, mocks, [cur, cur], 3, True)
    assert list(fruits) == mocks[-2:-1], list(remain) == mocks[-1:]
    fruits, remain = state_maintainer(11, mocks, [cur, cur], 2, True)
    assert list(fruits) == [] and list(remain) == mocks[-2:]
    fruits, remain = state_maintainer(11, mocks, [cur], 1, True)
    assert list(fruits) == [] and list(remain) == mocks[1:] and state_maintainer.__closure__[0].cell_contents == 1
    fruits, remain = state_maintainer(11, mocks, [cur], 0, False)
    assert list(fruits) == mocks[1:] and list(remain) == []
    fruits, remain = state_maintainer(11, mocks, [cur], 2, True)
    assert list(fruits) == mocks[1:2] and list(remain) == mocks[2:]


def test_state_slider():
    assert False


