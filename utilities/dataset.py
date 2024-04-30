from zipfile import ZipFile

import pandas as pd


def load_rounD(path, track_id):
    with ZipFile(f'{path}/rounD.zip') as zf:
        prefix = f'rounD/data/{track_id}_tracks'
        with zf.open(f'{prefix}Meta.csv') as f:
            meta = pd.read_csv(f, index_col='trackId', usecols=['trackId', 'class'])

        with zf.open(f'{prefix}.csv') as f:
            data = pd.read_csv(f)
    label = ('car', 'truck', 'van', 'bus', 'trailer', 'bicycle', 'motorcycle', 'pedestrian')
    class_map = meta['class'].replace({l: i for i, l in enumerate(label)})
    return class_map, data


def load_fake():
    import os
    file_name = os.path.join(os.path.dirname(__file__), '/home/lg/VDBM/spatiotemporal/resource/dataset/fake_tracks.csv')
    data = pd.read_csv(file_name)
    columns = ['oid', 'fid', 'x', 'y']
    data[['x', 'y']] *= 100
    return data, columns, data[['oid', 'cls']].drop_duplicates('oid').set_index('oid')['cls']


def load_yolo_for(filename):
    columns = ['oid', 'fid', 'x', 'y']
    data = pd.read_pickle(filename)
    return data, columns, data[['oid', 'cls']].drop_duplicates('oid').set_index('oid')['cls']
