import heapq
from collections import Counter, deque
from collections.abc import Iterable, MutableMapping, Callable
from collections.abc import Mapping, Sequence
from itertools import takewhile, islice
from operator import attrgetter

import numpy as np

from search.co_moving import CoMovementPattern
from utilities.trajectory import BasicTrajectorySeg


def yield_co_move(duration: int, labels: Mapping[int, int], active_space: MutableMapping[int, BasicTrajectorySeg],
                  timestamp: int, traj_required: Sequence) -> Iterable[tuple[list[int], int, int]]:
    """
    a co-movement checker respecting objects' label and count and their co-moving duration .
    Note that this function may modify `active_space`.
    :param duration: objects co-moving duration
    :param labels: {obj_label: count}
    :param active_space: {obj_id: trajectory}
    :param timestamp: current processing time
    :param traj_required: trajectories need processing
    :return: iterator of tuple(ids, start, end)
    """
    id_required = set(traj.id for traj in traj_required if timestamp - traj.begin >= duration)

    if not id_required:
        for traj in traj_required:
            del active_space[traj.id]
        return

    traj_cand = [traj for traj in takewhile(lambda traj: timestamp - traj.begin >= duration, active_space.values())]
    for traj in traj_required:
        del active_space[traj.id]

    result_bag = Counter(map(attrgetter('label'), traj_cand))
    ids = [traj.id for traj in traj_cand]

    # It's impossible for result bag to have more label types. because we filtered labels first.
    assert len(result_bag) <= len(labels)
    if len(result_bag) == len(labels):
        for tra_label in labels:
            if result_bag[tra_label] < labels[tra_label]:
                return
        res_pos = np.flatnonzero(np.diff(np.fromiter(map(attrgetter('begin'), traj_cand),
                                         np.int32, len(traj_cand)), append=-1))
        timestamp -= 1  # the end point is exclusive
        for i in res_pos[::-1]:  # start at the least duration
            traj = traj_cand[i]
            yield ids[:i+1], traj.begin, timestamp
            label = traj.label
            result_bag[traj.label] -= 1
            id_required.discard(traj.id)
            if not id_required or result_bag[label] < labels[label]:
                break


def group_until(queue, ts):
    if not queue or queue[0][0] > ts:
        return
    else:
        t, tid = heapq.heappop(queue)
        group = [tid]
        while queue:
            end = queue[0][0]
            if end == t:
                group.append(heapq.heappop(queue)[1])
            else:
                yield t, group
                if end > ts:
                    return
                t = end
                group = [heapq.heappop(queue)[1]]
        yield t, group


def sliding_window(df, win_len):
    df_iter = iter(df)
    window = deque(islice(df_iter, win_len), maxlen=win_len)
    if not window:
        return
    else:
        yield window
    for frame in df_iter:
        window.append(frame)
        yield window


def df_filter(df, reg_verifier, target_label):
    df = df[df['cls'].isin(target_label)]  # class verification
    df = df[reg_verifier(df[['x', 'y']])]  # region verification
    return df


def state_sliding(pat_series: Iterable[CoMovementPattern],
                  obj_verifier,
                  state_maintainer: Callable[[int, Sequence, Sequence, int, bool], [Iterable, Iterable]])\
        -> Iterable[CoMovementPattern]:
    # $prev stores seen patterns. Every pattern is a set of objs that co-moves a certain period
    # Note that in terms of object set, prev[0] ⊃ prev[1] ⊃ prev[2] ⊃ ...
    pat_iter = iter(pat_series)
    cur = next(pat_iter, None)
    if cur is None:
        return
    prev = [cur]
    for cur in pat_iter:
        absort = False
        cur_end = cur.end
        if cur.start > prev[0].end + 1:  # not consecutive in time interval
            count = len(prev)
            new = [cur]
        else:
            new = []
            count = 0
            for pat in prev:
                inter = pat.objs & cur.objs
                if len(inter) == len(pat):  # pat is a subset of cur
                    if len(inter) == len(cur):  # pat equals to cur
                        pat.end = cur_end
                    else:
                        new.append(cur)
                    absort = True
                    break

                elif len(inter) == len(cur):  # cur is a proper subset of pat
                    cur.start = pat.start
                    count += 1

                else:  # intersection is a new objet set
                    new_pattern = CoMovementPattern({obj: cur.labels[obj] for obj in inter})
                    new.append(cur)
                    if obj_verifier(new_pattern.label_count()):  # a new pattern
                        new_pattern.interval = [pat.start, cur_end]
                        cur = new_pattern
                        count += 1
                    else:
                        break
            else:
                new.append(cur)

        fruits, prev_iter = state_maintainer(cur_end, prev, new, count, absort)
        prev_end = prev[0].end
        for pat in fruits:
            pat.end = prev_end
            yield pat.to_plain()

        new.extend(prev_iter)
        prev = new

    if prev:
        fruits, _ = state_maintainer(0, prev, [], len(prev), False)
        prev_end = prev[0].end
        for pat in fruits:
            pat.end = prev_end
            yield pat.to_plain()
