from collections import defaultdict
from itertools import pairwise

import numpy as np

from indices.region import GridRegion
from indices.user_indices import get_user_indices
from utilities.trajectory import TrajectorySequenceSeg


def build_tempo_spatial_index(trajs, cfg):
    # 1. find trajectory's region
    # config.gird_border = gen_border(trajs.bbox, 10, 15)
    reg = GridRegion(cfg)
    UserIdx = get_user_indices(cfg)
    user_idx = defaultdict(UserIdx)

    for tid, beg, cls_id, track in trajs:
        pos = reg.index(track)
        break_points = np.flatnonzero(np.diff(pos, prepend=-1, append=-1))
        track_lifelong = len(track)
        spt_idx = user_idx[cls_id]
        for start, end in pairwise(break_points):
            ts_beg = start + beg
            spt_idx.add(((track[start], track_lifelong), (end - start, ts_beg)),
                        TrajectorySequenceSeg(tid, ts_beg, cls_id, track[start:end]))
    return user_idx


def save_trajectory_segments(trajs, cfg, save_path):
    import pickle
    reg = GridRegion(cfg)
    traj_l = []
    for tid, beg, cls_id, track in trajs:
        pos = reg.index(track)
        break_points = np.flatnonzero(np.diff(pos, prepend=-1, append=-1))
        track_lifelong = len(track)
        segments = []
        for start, end in pairwise(break_points):
            ts_beg = start + beg
            segments.append(TrajectorySequenceSeg(tid, ts_beg, cls_id, track[start:end]))
        traj_l.append((track_lifelong, segments))
    with open(save_path, 'wb') as f:
        pickle.dump(traj_l, f)
    return traj_l


def build_index_from_segments(segments, cfg):
    user_idx = defaultdict(get_user_indices(cfg))
    for lifelong, segs in segments:
        spt_idx = user_idx[segs[0].label]
        for s in segs:
            p = s.points
            spt_idx.add(((p[0], lifelong),
                         (len(p), s.begin)),
                        s)
    return user_idx
