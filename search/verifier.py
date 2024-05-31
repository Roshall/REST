from _bisect import bisect_right
from itertools import islice, pairwise
from operator import attrgetter
from typing import Iterable

import numpy as np
from utilities.box2D import Box2D
from traj_seg import TrajectoryIntervalSeg


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
        s = start_pos[0]
        if (sp_len := len(start_pos)) == 1:
            s_len = len(in_pos)
            if mask[0] or mask[-1] or s_len >= duration:
                s = in_pos[s] + begin
                yield TrajectoryIntervalSeg(sid, s, label, s_len)
            return

        s_len = start_pos[1] - s
        if mask[0] or s_len >= duration:
            s = in_pos[s] + begin
            yield TrajectoryIntervalSeg(sid, s, label, s_len)

        for s, e in pairwise(islice(start_pos, 1, sp_len)):
            if (s_len := e - s) >= duration:
                s = in_pos[s] + begin
                yield TrajectoryIntervalSeg(sid, s, label, s_len)

        s = start_pos[-1]
        s_len = len(in_pos) - s
        if mask[-1] or s_len >= duration:
            s = in_pos[s] + begin
            yield TrajectoryIntervalSeg(sid, s, label, s_len)


def obj_verify(target, label_map):
    if len(target) != len(label_map):
        return False
    for label in target:
        if target[label] > label_map[label]:
            return False
    return True


def len_filter(num_m, length):
    return [obj for obj in num_m if num_m[obj] >= length]


def df_filter(df, reg_verifier, target_label):
    df = df[df['cls'].isin(target_label)]  # class verification
    df = df[reg_verifier(df[['x', 'y']])]  # region verification
    return df


def label_verifier(label_counter):
    for count in label_counter.values():
        if count < 0:
            return False
    return True
