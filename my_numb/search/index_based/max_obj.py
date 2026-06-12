import heapq

import numpy as np
from numba import int32, types, njit, jit_module
from numba.experimental import jitclass
from numba.typed import Dict, List


# Define the dictionary type: dict[int, int]
dict_type = types.DictType(int32, int32)


entry_type = types.Tuple((int32, int32))

pat_type = types.Tuple((types.Set(int32), int32))

# begin, end, tid, label
# trajectory_type = int32[:]

def label_verifier(label_counter: dict_type):
    for count in label_counter.values():
        if count < 0:
            return False
    return True


def is_subset(keys_a: dict_type, keys_b: dict_type):
    # Check if all keys in `a` exist in `b`
    for key in keys_a:
        if key not in keys_b:
            return False
    return True


# @njit((types.ListType(pat_type), dict_type, dict_type, int32, int32),cache=True)
def concatenate(patterns, obj_m, label_count: dict_type, start, end_l):
    objs = set(obj_m)
    if not patterns:
        patterns.append((objs, start))
        return []
    elif start > end_l:
        fruits = patterns[:]
        patterns.clear()
        patterns.append((objs, start))
        return fruits

    old = patterns.copy()
    patterns.clear()
    count = 0
    adopt = False
    cur = objs
    label_count = label_count.copy()
    for p, ps in old:
        clen, plen = len(cur), len(p)

        if clen > plen:
            if cur > p: # ckeys > pkeys
                patterns.append((cur, start))
                adopt = True
                break
        elif clen < plen:
            if cur < p:
                start = ps
                count += 1
                continue
        else: #clen == plen
            if cur == p:
                adopt = True
                break

        patterns.append((cur, start))
        # update new pattern
        p_new = cur & p
        for o in cur - p_new:
            label_count[obj_m[o]] -= 1

        if label_verifier(label_count):
            start = ps
            cur = p_new
            count += 1
        else:
            break
    else:
        patterns.append((cur, start))

    p_end = count if adopt else len(old)
    fruits = old[:p_end]
    for i in range(p_end, len(old)):
        patterns.append(old[i])
    return fruits

# @njit((types.ListType(tuple_type), int32), cache=True)
# def pop_all(patterns, end):
#     for pat, start in patterns:
#         yield pat, start, end
#     patterns.clear()

def groupby(iterable, pos):
    # [k for k, g in groupby('AAAABBBCCDAABBB')] → A B C D A B
    # [list(g) for k, g in groupby('AAAABBBCCD')] → AAAA BBB CC D

    iterator = iter(iterable)
    for curr_value in iterator:
        break
    else:
        return
    curr_key = curr_value[pos]
    group = [curr_value]

    target_key = curr_key
    for curr_value in iterator:
        curr_key = curr_value[pos]
        if curr_key != target_key:
            yield curr_key, group
            group = [curr_value]
        else:
            group.append(curr_value)

def update_state(trajs, label_m, label_counter, end_q):
    for tra in trajs:
        tid = tra[1]
        label = tra[2]
        label_counter[label] += np.int32(1)
        label_m[tid] = label
        heapq.heappush(end_q, (tra[3], tid))

def init_state(start, dur, ts_grouped_traj, label_m, label_counter, end_q):
        for ts, trajs in ts_grouped_traj:
            if ts >= start:
                update_state(trajs, label_m, label_counter, end_q)
                return ts
            else:
                min_end = start + dur
                update_state([tra for tra in trajs if tra[1] >= min_end], label_m, label_counter, end_q)
        else:
            if end_q:
                return start
            else:
                return -1

def heap_group_pop(heap):
    if heap:
        n = heap[0][0]
    else:
        return
    while heap and heap[0][0] == n:
        yield heapq.heappop(heap)[1]

def remove_traj(label_counter, label_m, end_q):
    for tid in heap_group_pop(end_q):
        label_counter[label_m[tid]] -= 1
        del label_m[tid]

def walk_through(trajectories, labels, dur, start, terminal_t):
    grouped_traj = groupby(trajectories, 0)
    final_s = terminal_t - dur
    label_counter = {label: -count for label, count in labels.items()}
    label_m = Dict.empty(entry_type)
    end_q = List.empty_list(entry_type)
    last_s = init_state(start, dur, grouped_traj, label_m, label_counter, end_q)
    if  last_s == -1:
        return
    ppool = List.empty_list(pat_type)
    pool_end = 0
    for ts, trajs in grouped_traj:
        if ts <= final_s:
            end_min = ts + dur
            while end_q:  # in case that all objects have gone
                # we want to concatenate the windows, so the last window end (end_q[0][0])
                # must >= ts + dur_l.
                end = end_q[0][0]
                if end < end_min:
                    if label_verifier(label_counter):
                        for objs, s in concatenate(ppool, label_m, label_counter, last_s, pool_end):
                            yield objs, s, pool_end
                        pool_end = end
                        remove_traj(label_counter, label_m, end_q)
                    else:
                        remove_traj(label_counter, label_m, end_q)
                        # impossible to concatenate
                        for objs, start in ppool:
                            yield objs, start, pool_end
                        while end_q and end_q[0][0] < end_min:
                            # some objects having multiple trajectories,
                            # this will ensure we first remove the old trajectory
                            remove_traj(label_counter, label_m, end_q)
                        break  # no need to find end >= ts
                else:  # found the end >= ts + dur_l, construct the final window, and get out of the loop
                    if label_verifier(label_counter):
                        for objs, start in concatenate(ppool, label_m, label_counter, last_s, pool_end):
                            yield objs, start, pool_end
                        pool_end = end
                    else:
                        # impossible to concatenate
                        for objs, start in ppool:
                            yield objs, start, pool_end
                    break
            last_s = ts
            update_state(trajs, label_m, label_counter, end_q)
        else:
            break

    # wrapup
    while end_q:
        if label_verifier(label_counter):
            end = end_q[0][0]
            if end < terminal_t:
                for objs, s in concatenate(ppool, label_m, label_counter, last_s, end):
                    yield objs, s, pool_end
                    pool_end = end
                    remove_traj(label_counter, label_m, end_q)
            else:
                for objs, s in concatenate(ppool, label_m, label_counter, last_s, terminal_t):
                    yield objs, s, pool_end
                    pool_end = terminal_t
                    break
        else:
            break

    max_start = pool_end - dur
    for objs, start in ppool:
        if start <= max_start:
            yield objs, start, pool_end
        else:
            break

# jit_module(nopython=True, cache=True)

@njit
def t_gen():
    for i in range(10):
        yield np.int32(i), np.int32(i+100), np.int32(i+200), np.int32(i+300), np.arange(i, i+4, dtype=np.float32)
        if i % 2 == 0:
            yield np.int32(i), np.int32(i+10), np.int32(i+20), np.int32(i+30), np.arange(i, i+4, dtype=np.float32)

@njit
def t_groupby():
    for k, g in groupby(t_gen(), 0):
        print(k)
        count = 0
        for obj in g:
            count += 1
            print(obj)
        print(count)

@njit
def t_update_state():
    trajs = [(1, 1, 1, 1)]
    label_counter = {np.int32(1): np.int32(1)}
    label_m = {np.int32(0): np.int32(1)}
    end_q = [(0, 10)]
    update_state(trajs, label_m, label_counter, end_q)
    print(label_counter, label_m, end_q)


def max_obj_num_enumerator_jit(trajectories, labels, dur, interval):
    return walk_through(trajectories, labels, dur, *interval)



if __name__ == '__main__':
    # obj_m = Dict.empty(key_type=int32, value_type=int32)
    # obj_m[1] = 1
    # obj_m[2] = 2
    # label_count = Dict.empty(key_type=int32, value_type=int32)
    # label_count[1] = 1
    # label_count[2] = 1
    # ppool = List.empty_list(tuple_type)
    # ppool.append((obj_m, np.int32(0)))
    # end = 1
    # concatenate(ppool, end, obj_m, label_count, np.int32(0))
    t_groupby()

