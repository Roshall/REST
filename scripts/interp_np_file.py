from collections import defaultdict
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from utilities.data_preprocessing import traj_interp, centering_sort_by_oid


def comb_file(fs):
    if len(fs) > 1:
        f_ls = sorted(fs)
        f_ls.append(f_ls.pop(0))
        arr_ls = [np.load(f, mmap_mode='r') for f in f_ls]
        return np.concatenate(arr_ls)
    else:
        return np.load(fs[0])


def try_comb(vdir):
    cob_map = defaultdict(lambda: defaultdict(list))
    for f in Path(vdir).glob('*.npy'):
        prefix, dast = f.stem.split('_')[:2]
        cob_map[dast][prefix].append(f)
    for dast, prefixes in cob_map.items():
        yield dast, (comb_file(fs) for fs in prefixes.values())


if __name__ == '__main__':
    # video_dir = "/home/lg/VDBM/spatiotemporal/resource/new_florida5h.npy"
    # out = '/home/lg/VDBM/spatiotemporal/resource/'
    chunk_th = 8 << 30
    if len(sys.argv) < 3:
        raise ValueError('too few arguments, please provide input file and output path')
    video_dir, out = sys.argv[1:3]
    # trans_map = {'beachbar': False, 'florida': False, 'jacksontown': False,
    #              'ireland': False, 'osaka': False, 'shinjuku': True,
    #              'timesquare': True}
    # files = sorted(Path(video_dir).glob('*.npy'))
    data_partition = try_comb(video_dir)
    # for human, others in zip(files, files[len(files)//2:]):
    for ds, partition in data_partition:
        print(f'loading {ds}...')
        dfs = []
        max_id = -1
        for part in partition:
            if max_id < 0:
                max_id = part[:, 0].max()
            else:
                part[:, 0] += max_id
            print(f'interpolating {ds}...')
            if part.nbytes > chunk_th:
                oid_sorted = centering_sort_by_oid(part, False)
                del part
                df = traj_interp(oid_sorted, chunk_size=chunk_th)
                del oid_sorted
            else:
                df = traj_interp(part, False)
                del part
            dfs.append(df)

        print(f'concatenating {ds} ...')
        df = pd.concat(dfs, ignore_index=True)
        if df.memory_usage(index=True).sum() < 2 << 30:
            df.to_pickle(os.path.join(out, f'{ds}.pkl'))
        else:
            df.to_parquet(os.path.join(out, f'{ds}.parquet'))
