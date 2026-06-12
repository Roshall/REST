import cppyy
from numba import njit, jit_module

from my_numb.search.verifier import candidate_verified
from my_numb.utilities.iterate_tools import trajectory_manger_points, trajectory_manger_start_end

def absorb(visited, dur):
    trajs = []
    for tid, label, seq in visited.values():
        if len(seq) > 2:
            seq.sort()
            beg = seq[0]
            for i in range(1, len(seq) - 1, 2):
                prev_end, succ_start = seq[i], seq[i + 1]
                if prev_end != succ_start:
                    if prev_end - beg >= dur:
                        trajs.append((beg, prev_end - 1, tid, label))
                    beg = succ_start
            if seq[-1] - beg >= dur:
                trajs.append((beg, seq[-1] - 1, tid, label))
        else:
            if seq[-1] - seq[0] >= dur:
                trajs.append((seq[0], seq[-1] - 1, tid, label))
    return trajs

def chain(a, b):
    for x in a:
        yield x
    for x in b:
        yield x
def vanilla_merge(rest_query, trajs_meta, region, labels, dur, interval):
    visited = dict()
    region_meta = labels.reshape(-1, 2).T.copy()
    for c in labels:
        rest_query(c, region, dur, interval, 1)
        for beg, end, tid, label in chain(candidate_verified(trajectory_manger_points(trajs_meta, 0), region_meta, dur),
                                          trajectory_manger_start_end(trajs_meta, 1)):
            if tid not in visited:
                visited[tid] = (tid, label, [beg, end])
            else:
                visited[tid][2].extend([beg, end])

    trajs = absorb(visited, dur)
    trajs.sort()
    return trajs

# jit_module(nopython=True, cache=True)