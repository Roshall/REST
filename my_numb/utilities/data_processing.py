import numpy as np


def traj_df2np(df, cols_name, cls=None):
    """
    Convert trajectory dataframe to numpy array.
    :param df: pandas dataframe of trajectory data
    :param cols_name: required columns name of the dataframe, [track_id, frame_id, x, y]
    :param cls: remaining classes to be included in the output
    :return: tuple of (structure, points), where structure is (begins, ids, labels, offsets, s_offsets(empty currently))
    """
    cols_name.append('cls')
    if cls is not None:
        df = df[df['cls'].isin(cls)]
    df = df[cols_name]
    df = df.sort_values(by=cols_name[:2])
    oids = df[cols_name[0]].to_numpy()
    offsets = np.flatnonzero(np.diff(oids, prepend=-1, append=-1))
    beg_id_cls_pos_sos = np.empty((len(offsets), 5), dtype=np.int32)
    beg_id_cls_pos_sos[:-1, :3] = df[[cols_name[1], cols_name[0], cols_name[-1]]].to_numpy()[offsets[:-1]]
    beg_id_cls_pos_sos[:, 3] = offsets
    points = df[cols_name[-3:-1]].to_numpy(copy=True)
    return beg_id_cls_pos_sos, points


if __name__ == '__main__':
    from utilities.dataset import load_fake

    data, cols = load_fake(False)
    package = traj_df2np(data, cols)
    print(package)
