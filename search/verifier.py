from _bisect import bisect_right
from itertools import islice
from operator import attrgetter
from typing import Iterable

import numpy as np
from utilities.box2D import Box2D
from utilities.trajectory import TrajectoryIntervalSeg

def candidate_verified_queue(candidates: Iterable, region: Box2D, duration: int) -> Iterable[TrajectoryIntervalSeg]:
    """
    verify trajectories and find the segments within region
    :param region: a box object with `enclose` function implemented
    :param candidates: trajectories source
    :param duration: the least lifetime of a segment
    :return: trajectory segments within region, which are guaranteed to be sorted by `begin` if `candidates` is sorted.
    """
    verified = []
    for cand_seg in candidates:
        mask = region.enclose(cand_seg.points)

        pos = bisect_right(verified, cand_seg.begin, key=attrgetter('begin'))
        if pos > 0:
            yield from islice(verified, pos)
            verified = verified[pos:]
        verified.extend(verify_seg(cand_seg, mask, duration))
        verified.sort(key=attrgetter('begin'))
    yield from verified


def verify_seg(segment, mask, duration: int):
    """
    verify a trajectory segment, and find parts within region
    :param segment: trajectory sequence segment.
    :param region: a box object with `enclose` function implemented
    :param duration: the least lifetime of a segment
    :return: sorted parts of the segment by `begin`simplify verified queue
    """
    in_pos = np.flatnonzero(mask)
    if len(in_pos) != 0:
        sid, begin, label = segment.id, segment.begin, segment.label
        if mask.all():
            yield TrajectoryIntervalSeg(sid, begin, label, len(mask))
            return
        tmp = np.empty(in_pos.shape[0]+1, dtype=np.int32)
        tmp[0] = -2
        tmp[1:] = in_pos
        start_pos = np.flatnonzero(np.diff(tmp) > 1)
        tmp = tmp[:start_pos.shape[0]+1]
        tmp[:-1] = start_pos
        tmp[-1] = len(in_pos)
        seg_lens = np.diff(tmp)
        res_mask = np.flatnonzero(seg_lens >= duration)
        if mask[0] and seg_lens[0] < duration:
            yield TrajectoryIntervalSeg(sid, begin, label, seg_lens[0])

        for m in res_mask:
            yield TrajectoryIntervalSeg(sid, begin + in_pos[start_pos[m]], label, seg_lens[m])

        if mask[-1] and seg_lens[-1] < duration:
            yield TrajectoryIntervalSeg(sid, begin + in_pos[start_pos[-1]], label, seg_lens[-1])


def obj_verify(target, label_map):
    if len(target) != len(label_map):
        return False
    for label in target:
        if target[label] > label_map[label]:
            return False
    return True


def len_filter(num_m, length):
    return [obj for obj in num_m if num_m[obj] >= length]
