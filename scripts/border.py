import os.path
from typing import Iterable

import numpy as np

from utilities.data_preprocessing import view_field


def life_border_cls_top(df, quantiles: tuple = ([0, 5000, 50000, 10000], [0.5, 0.2, 0.1, 0.05]), k: int = 9):
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
    reg_border_by_cls = {}
    for c in cls:
        df_cls = df.loc[df.cls == c]
        reg_border_by_cls[c] = list(map(int, view_field(df_cls[['x', 'y']])))
    return reg_border_by_cls


def border_meta(df, tempo_stride=60):
    border = {}
    border_by_cls = life_border_cls_top(df)
    reg_border_by_cls = region_border(df, border_by_cls)
    for c, life_border in border_by_cls.items():
        border[c] = {'life_border': life_border, 'reg_border': reg_border_by_cls[c],
                     'tempo_stride': tempo_stride}
    return border


def border_stride(border, space):
    border = np.asarray(border).reshape(-1, 2)
    stride = (border[:, 1] - border[:, 0] - 1) // np.asarray(space) + 1
    bor_st = list(border[0])
    bor_st.append(stride[0])
    bor_st.extend(border[1])
    bor_st.append(stride[1])
    return bor_st


def load_ext_index_meta(fname):
    import json
    with open(fname) as f:
        meta = json.load(f)
    return {int(k): v for k, v in meta.items()}


if __name__ == '__main__':
    from utilities import dataset
    import json
    from configs import cfg
    path = cfg.DATA.PATH
    dataset_name = 'shinjuku3h'
    jsonpath = os.path.join(cfg.INDEX.META_PATH, dataset_name + '.json')
    if not os.path.exists(jsonpath):
        filepath = os.path.join(path, f'{dataset_name}.pkl')
        df, _, _ = dataset.load_yolo_for(filepath)
        border = border_meta(df)

        json.dump(border, open(jsonpath, 'w'), indent=4)
    else:
        print(f'File {jsonpath} exists')


