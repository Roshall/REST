from collections import defaultdict
from itertools import pairwise

import numpy as np

from indices.region import GridRegion
from indices.user_indices import get_user_indices
from utilities.trajectory import TrajectorySequenceSeg


def build_tempo_spatial_index(trajs, cfg):
    # 1. find trajectory's region
    # config.gird_border = gen_border(trajs.bbox, 10, 15)
    reg = GridRegion()
    UserIdx = get_user_indices(cfg)
    user_idx = defaultdict(UserIdx)

    for tid, beg, cls_id, track in trajs:
        pos = reg.index(track)
        break_points = np.flatnonzero(np.diff(pos, prepend=-1, append=-1))
        spt_idx = user_idx[cls_id]
        track_lifelong = len(track)
        for start, end in pairwise(break_points):
            ts_beg = start + beg
            spt_idx.add(((track[start], track_lifelong), (end - start, ts_beg)),
                        TrajectorySequenceSeg(tid, ts_beg, cls_id, track[start:end]))

    return user_idx
