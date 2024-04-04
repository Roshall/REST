from functools import partial

from search.base import base_maintainer
from search.co_moving import CoMovementPattern
from search.rest import state_sliding, df_filter
from search.verifier import obj_verify, len_filter
from search.baseline.naive import NaiveSliding
from utilities.box2D import Box2D
from utilities.data_preprocessing import group_by_frame
from utilities.dataset import load_fake


class TestSliding:
    data, _, _ = load_fake()
    interval = 0, 10
    frames = group_by_frame(data, interval)
    region = Box2D((0, 400, 0, 500))
    labels = {0: 1}
    duration = 3
    obj_verifier = partial(obj_verify, labels)
    dfilter = partial(df_filter, reg_verifier=region.enclose, target_label=labels.keys())

    def test_naive(self):
        sliding = NaiveSliding(self.frames, self.duration, self.obj_verifier, self.dfilter)
        res = list(sliding)
        assert len(res) == 8

    def test_expend(self):
        ans = [CoMovementPattern(labels={1: 0, 3: 0, 2: 0}, interval=[5, 10]),
               CoMovementPattern(labels={1: 0, 3: 0}, interval=[4, 10]),
               CoMovementPattern(labels={1: 0}, interval=[1, 10])]
        sliding = NaiveSliding(self.frames, self.duration, self.obj_verifier, self.dfilter)
        res = list(state_sliding(sliding, self.obj_verifier, base_maintainer))

        assert res == [pat.to_plain() for pat in ans]

