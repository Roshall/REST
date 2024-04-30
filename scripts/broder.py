import os.path
from typing import Iterable

import numpy as np

from utilities.data_preprocessing import view_field


def life_broder_cls_top(df, quantiles: tuple = ([0, 5000, 50000, 10000], [0.5, 0.2, 0.1, 0.05]), k: int = 9):
    df = df.drop(columns=['x', 'y', 'fid'])
    filtered = df.drop_duplicates().groupby('cls').count().sort_values(by='oid', ascending=False)[:k]
    group_cls = df.groupby('cls')
    broder_by_cls = {}
    for cls in filtered.index:
        duration = df.iloc[group_cls.indices.get(cls)].groupby('oid').count().sort_values(by='oid')
        q = quantiles[1][np.searchsorted(quantiles[0], duration.size, 'right')-1]
        space = np.linspace(0, 1-q, int(1/q))
        broder_by_cls[cls] = list(duration.quantile(space).astype(int)['cls'])
    return broder_by_cls


def region_border(df, cls: Iterable[int]):
    reg_broder_by_cls = {}
    for c in cls:
        df_cls = df.loc[df.cls == c]
        reg_broder_by_cls[c] = list(map(int, view_field(df_cls[['x', 'y']])))
    return reg_broder_by_cls


def border_meta(df, tempo_stride=60):
    broder = {}
    broder_by_cls = life_broder_cls_top(df)
    reg_broder_by_cls = region_border(df, broder_by_cls)
    for c, life_border in broder_by_cls.items():
        broder[c] = {'life_border': life_border, 'reg_broder': reg_broder_by_cls[c],
                     'tempo_stride': tempo_stride}
    return broder


if __name__ == '__main__':
    from utilities import dataset
    import json
    from configs import cfg
    path = '/home/lg/VDBM/spatiotemporal/regional_tempo_spatial_query/test'
    filename = 'timsquare3h.pkl'
    filepath = os.path.join(path, filename)
    df, _, _ = dataset.load_yolo_for(filepath)
    border = border_meta(df)
    filepath = os.path.join(cfg.INDEX.CONFIG_PATH, filename.split('.')[0] + '.json')
    json.dump(border, open(filepath, 'w'), indent=4)
