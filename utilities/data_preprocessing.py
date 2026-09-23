from collections import UserList

import numpy as np
import pandas as pd


def view_field(xy):
    return np.vstack((xy.min(), xy.max())).T.flatten()


def traj_data(tracks, cols_name: list, label_map, stride, scale=100, cls=None):
    """
    load objects data to trajectory entry.
    :param tracks: pandas data frame of tracks
    :param cols_name: [track_id, frame_id, x, y] are wanted, supply their actual properties name in order.
    :param stride: integer
    :param label_map: map traj id to its label
    :param scale:  scale raw (x, y).
    :param cls: which classes we want to use
    :return: generate of TrajTrace.
    """
    if cls is not None:
        tracks = tracks[tracks['cls'].isin(cls)]
    tracks = tracks[cols_name]
    traces_by_id = tracks.groupby(cols_name[0])
    for tid, t in traces_by_id:
        cls_id = label_map[tid]
        t = t.sort_values(by='fid')
        start_frame = t[cols_name[1]].iat[0]
        if stride > 1:
            trajs = t[cols_name[-2:]][::stride].to_numpy(copy=True)
        else:
            trajs = t[cols_name[-2:]].to_numpy(copy=True)
        if scale != 1:
            trajs *= scale
        yield tid, start_frame, cls_id, trajs


def gen_border(bbox, xy_num):
    xmin, xmax, ymin, ymax = bbox
    x_num, y_num = xy_num
    x_series = np.linspace(xmin, xmax, x_num, dtype=int)
    x_series = np.append(x_series, x_series[-1] + 1)
    y_series = np.linspace(ymin, ymax, y_num, dtype=int)
    y_series = np.append(y_series, y_series[-1] + 1)
    return x_series, y_series


def draw_traj_point_in_grid(data, reg_borders):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    fig, axs = plt.subplots()
    axs.scatter(*data.T)
    axs.grid()
    axs.xaxis.set_major_locator(FixedLocator(reg_borders[0]))
    axs.yaxis.set_major_locator(FixedLocator(reg_borders[1]))
    plt.show()


def group_by_frame(df, interval):
    df = df.sort_values(by='fid')  # in case that groups are not sorted by fid
    # Half-open interval [lo, hi): a record at fid == hi is excluded, matching
    # the canonical co-movement contract (mock_cases.clip, the index path, and
    # the C++ GroupByFrame).
    df = df.query(f'{interval[0]} <= fid < {interval[1]}')
    return df.groupby('fid')


def chunks(chunk_size, num):
    chunk_series = []
    while num > chunk_size:
        chunk_series.append(chunk_size)
        num -= chunk_size
    if num != 0:
        chunk_series.append(num)
    return chunk_series


class SingleRun(UserList):
    def __init__(self, header, dtypes):
        super().__init__()
        self.header = header
        self.dtypes = dtypes

    def merge(self):
        run = np.concatenate(self)
        return (pd.DataFrame(run, columns=self.header).
                astype(dict(zip(self.header, self.dtypes))))


class ChunkRun:
    def __init__(self, chunk_size, header, dtypes):
        self.merge_ls = []
        self.interp_ls = SingleRun(header, dtypes)
        self.chunk_size = chunk_size
        self.csize = 0
        self.header = header

    def append(self, item):
        if self.csize >= self.chunk_size:
            self.merge_ls.append(self.interp_ls.merge())
            self.interp_ls.clear()
            self.csize = 0
        self.interp_ls.append(item)
        self.csize += item.nbytes

    def merge(self):
        ml = self.merge_ls
        if self.interp_ls:
            ml.append(self.interp_ls.merge())
        if len(ml) > 1:
            return pd.concat(ml, ignore_index=True)
        else:
            return ml[0]


def run_factory(chunk_size, header, dtypes):
    if chunk_size is None:
        return SingleRun(header, dtypes)
    else:
        return ChunkRun(chunk_size, header, dtypes)


def centering_sort_by_oid(np_arr, center):
    np_arr[:, 3] += np_arr[:, 5]
    np_arr[:, 3] //= 2
    if center:
        np_arr[:, 6] += np_arr[:, 4]
        np_arr[:, 6] //= 2

    # sort by oid
    return np_arr[np.argsort(np_arr[:, 0]).reshape(-1, 1), [*range(4), 6]]


def traj_interp(np_arr, center=True, chunk_size=None):
    if chunk_size is None:
        np_arr = centering_sort_by_oid(np_arr, center)
    header = ['oid', 'cls', 'fid', 'x', 'y']
    # group by objects
    diff = np.flatnonzero(np.diff(np_arr[:, 0], prepend=np_arr[0, 0]))
    group_by_id = np.split(np_arr, diff)
    id_num = np.flatnonzero(np.diff(diff) > 1).size
    del diff
    runner = run_factory(chunk_size, header, (np.int32, np.uint8, np.int32, np.uint16, np.uint16))
    count = 0
    for group in group_by_id:
        if len(group) == 1:
            continue
        group = group[np.argsort(group[:, 2])]
        fids = group[:, 2]

        min_, max_, = fids[[0, -1]]
        vals, counts = np.unique(group[:, 1], return_counts=True)
        mod = vals[np.argmax(counts)]  # mode as revised class id
        if len(group) != max_ - min_ + 1:  # interpolate
            interp_fids = np.arange(min_, max_ + 1)
            interp = np.empty((len(interp_fids), len(header)), dtype=np.int32)
            interp[:, 0] = group[0, 0]
            interp[:, 2] = interp_fids
            for col in range(3, 5):
                interp[:, col] = np.interp(interp_fids, fids, group[:, col])
        else:
            interp = group
        interp[:, 1] = mod  # revise class id
        runner.append(interp)
        count += 1
        if count % 8000 == 0:
            print(f'\r{count}/{id_num}', end='')
    del group, group_by_id, np_arr
    print(f'\rinterpolation done. merging into pandas DataFrame...')
    return runner.merge()


def debug_inter():
    data = np.arange(32).reshape(-1, 5)


if __name__ == '__main__':
    from utilities import dataset

    # file_path = '/home/lg/VDBM/spatiotemporal/resource/dataset'
    # cls_map, data = dataset.load_rounD(file_path, '00')
    # cols = ['trackId', 'frame', 'xCenter', 'yCenter']
    filename = '/media/cw/DataSet/Dataset/detection_results/order_florida5h.pkl'
    fps = 30
    data, cols, cls_map = dataset.load_yolo_for(filename)
    XY = data[cols[-2:]]
    XY = XY[::30]
    draw_traj_point_in_grid(XY, gen_border(view_field(XY), (10, 15)))
