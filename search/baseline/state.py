from collections.abc import Iterable, Sequence
from itertools import islice

from search.co_moving import CoMovementPattern
from search.rest import state_sliding


def state_slider(frames: Iterable, win_len, obj_verifier, dfilter):
    state_maintainer = state_maintain(win_len)
    pat_iter = pat_wrapper(frames, obj_verifier, dfilter)
    return state_sliding(pat_iter, obj_verifier, state_maintainer)


def pat_wrapper(frames, obj_verifier, dfilter):
    for fid, objs in frames:
        objs = dfilter(objs)
        objs = objs.set_index('oid')['cls'].to_dict()['cls']
        if obj_verifier(objs.keys()):
            yield CoMovementPattern(objs, [fid, fid])


def state_maintain(win_len):
    valid_ptr: int = 0

    def inner(end, prev: Sequence, new: Sequence, count: int, absort: bool):
        nonlocal valid_ptr
        prev_iter = iter(prev)
        # fruits is in [valid_ptr, count) if absort or [valid_ptr, len(prev))
        # in case that count <= valid_ptr, no fruit can be yielded, but we must only advance iterator to `count`
        # that's why min(valid_ptr, count)
        fruits = islice(prev_iter, min(valid_ptr, count), count if absort else None)
        if absort and count < valid_ptr:
            if win_len - (end - prev[valid_ptr - 1].start) < 1:  # check if we can recede pointer
                valid_ptr -= 1
            valid_ptr += len(new) - count
        else:
            valid_ptr = len(new)
        return fruits, prev_iter

    return inner
