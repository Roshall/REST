from collections import Counter, deque
from functools import partial
from itertools import islice

from search.index_based.base import base_maintainer
from search.co_moving import CoMovementPattern
from search.rest import state_sliding
from search.verifier import len_filter


class NaiveSliding:
    def __init__(self, frames, win_len, obj_ver, dfilter):
        self.frames = frames
        self.olen_m, self.label_m = Counter(), {}
        self.len_filter = partial(len_filter, self.olen_m, win_len)
        self.obj_ver = obj_ver
        self.dfilter = dfilter
        self.win_len = win_len

    def _update(self, objs):
        objs = self.dfilter(objs)
        self.olen_m.update(objs['oid'])
        self.label_m.update(objs.set_index('oid')['cls'])

    def _filter(self, low, high):
        ids = self.len_filter()
        if ids:
            candi = CoMovementPattern({id_: self.label_m[id_] for id_ in ids})
            if self.obj_ver(candi.label_count()) and high - low == self.win_len - 1:
                candi.interval = [low, high]
                return candi
            else:
                return False
        else:
            return False

    def _subtract(self, abandoned):
        self.olen_m.subtract(abandoned)
        desolated = [obj for obj, num in self.olen_m.items() if num <= 0]
        for obj in desolated:
            del self.olen_m[obj]
            del self.label_m[obj]

    def __iter__(self):
        frames_iter = iter(self.frames)
        win_len = self.win_len
        win = deque(maxlen=win_len)
        for fid, objs in islice(frames_iter, win_len):
            objs = self.dfilter(objs)
            win.append((fid, objs))
            self._update(objs)
        low, high = win[0][0], win[-1][0]
        if candi := self._filter(low, high):
            yield candi

        abandoned = win[0][1]['oid']
        for fid, objs in frames_iter:
            objs = self.dfilter(objs)
            win.append((fid, objs))
            self._subtract(abandoned)
            self._update(objs)

            low = win[0][0]
            if candi := self._filter(low, fid):
                yield candi
            abandoned = win[0][1]['oid']


def naive_slider(frames, dur, obj_verifier, dfilter):
    sliding = NaiveSliding(frames, dur, obj_verifier, dfilter)
    return state_sliding(sliding, obj_verifier, base_maintainer)
