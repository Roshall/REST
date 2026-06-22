from collections import Counter
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
        if not objs.empty:
            objs = objs.set_index("oid")["cls"].to_dict()
            if obj_verifier(Counter(objs.values())):
                yield CoMovementPattern(objs, [fid, fid])


def state_maintain(win_len):
    valid_ptr: int = 1
    win_len -= 1  # end - start = w_Len = 1

    def inner(prev_end, prev: Sequence, new_len: int, count: int, absorb: bool):
        nonlocal valid_ptr
        prev_iter = iter(prev)
        # check if we can move prev's valid pointer upwards
        least_start = prev_end - win_len
        while valid_ptr != 0 and least_start >= prev[valid_ptr - 1].start:
            valid_ptr -= 1
        # fruits is in [valid_ptr, count) if absorb or [valid_ptr, len(prev))
        # in case that count <= valid_ptr, no fruit can be yielded, but we must only advance iterator to `count`
        # that's why min(valid_ptr, count)
        if absorb:
            fruits = islice(prev_iter, min(valid_ptr, count), count)
            if count < valid_ptr:
                valid_ptr += new_len - count
            else:
                valid_ptr = new_len
        else:
            fruits = islice(prev_iter, valid_ptr, None)
            valid_ptr = new_len
        return fruits, prev_iter

    return inner
