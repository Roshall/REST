import numpy as np


def view_field(object_tb, x_name, y_name, scale=1):
    X = object_tb[x_name]
    Y = object_tb[y_name]
    return np.array((X.min(), X.max(), Y.min(), Y.max()), dtype=np.int32) * scale


def traj_data(tracks, cols_name: list, fps, label_map, scale=100):
    """
    load objects data to trajectory entry.
    :param tracks: pandas data frame of tracks
    :param cols_name: [track_id, frame_id, x, y] are wanted, supply their actual properties name in order.
    :param fps: integer
    :param label_map: map traj id to its label
    :param scale: the unit of our coordinate is 'cm', tell us how we scale raw (x, y).
    :return: trajectories tuple with format ('TrajTrace', 'tId start_frame track') and points of all trajectories.
    """
    tracks = tracks[cols_name]
    traces_by_id = tracks.groupby(cols_name[0])
    for tid, t in traces_by_id:
        cls_id = label_map[tid]
        start_frame = t[cols_name[1]].iat[0]
        trajs = (t[cols_name[-2:]][::fps].to_numpy() * scale).astype(np.int32)
        yield tid, start_frame,  cls_id,  trajs


def gen_border(bbox, x_num, y_num):
    xmin, xmax, ymin, ymax = bbox
    x_series = np.linspace(xmin, xmax, x_num, dtype=int)
    x_series = np.append(x_series, x_series[-1]+1)
    y_series = np.linspace(ymin, ymax, y_num, dtype=int)
    y_series = np.append(y_series, y_series[-1]+1)
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
    df = df.query(f'{interval[0]} <= fid <= {interval[1]}')
    return df.groupby('fid')


def traj_interp(np_arr, center=True):
    import pandas as pd

    np_arr[:, 3] += np_arr[:, 5]
    np_arr[:, 3] //= 2
    if center:
        np_arr[:, 4] += np_arr[:, 6]
        np_arr[:, 4] //= 2

    header = ['oid', 'cls', 'fid', 'x', 'y']
    # sort by oid
    np_arr = np_arr[np.argsort(np_arr[:, 0]), :5]
    # group by objects
    diff = np.flatnonzero(np.diff(np_arr[:, 0], prepend=np_arr[0, 0]))
    group_by_id = np.split(np_arr, diff)
    id_num = len(group_by_id)
    print('total #id:', id_num, '\n')
    interp_ls = [None] * id_num
    for i, group in enumerate(group_by_id):
        fids = group[:, 2]
        min_, max_, = np.min(fids), np.max(fids)
        vals, counts = np.unique(group[:, 1], return_counts=True)
        mod = vals[np.argmax(counts)]  # mode as revised class id
        if len(group) != max_ - min_ + 1:  # interpolate
            interp_fids = np.arange(min_, max_+1)
            interp = np.empty((len(interp_fids), len(header)), dtype=np.int32)
            interp[:, 0] = group[0, 0]
            interp[:, 2] = interp_fids
            for col in range(3, 5):
                interp[:, col] = np.interp(interp_fids, fids, group[:, col])
        else:
            interp = group
        interp[:, 1] = mod  # revise class id

        interp_ls[i] = interp
        if i % 1000 == 0:
            print(f'\r{i}/{id_num}', end='')
    print('merging...')
    merged = np.concatenate(interp_ls)
    return pd.DataFrame(merged, columns=header).astype(dict(zip(header, (np.int32, np.uint8, np.int32, np.uint16, np.uint16))))


if __name__ == '__main__':
    from utilities import dataset
    import os

    file_path = '/home/lg/VDBM/spatiotemporal/resource/dataset'
    fps = 25
    cls_map, data = dataset.load_rounD(file_path, '00')
    cols = ['trackId', 'frame', 'xCenter', 'yCenter']
    raw_traj = traj_data(data, cols, fps, cls_map)
    XY = data[cols[-2:]].to_numpy()
    draw_traj_point_in_grid(XY, gen_border(raw_traj.bbox, 10, 15))
