import os
from functools import partial

from ext_index.builder import load_traj_seg, build_rest
from scripts.border import load_ext_index_meta
from search.index_based.framework import index_based_framework
from search.sliding_based.framework import sliding_framework


def build_index(fname, ds_name, config):
    bor_m = load_ext_index_meta(os.path.join(config.INDEX.META_PATH,
                                             f'{ds_name}.json'))
    trajs, segs = load_traj_seg(bor_m, fname, ds_name, config)
    spt_idx = build_rest(trajs, segs, bor_m)
    return [spt_idx, trajs]


def query_init(query_type, trajs_info):
    query_type = query_type.split('_')
    match query_type:
        case ('index', *mtd_str):
            search_mtd = partial(index_based_framework, method=''.join(mtd_str))
        case ('sliding', *mtd_str):
            search_mtd = partial(sliding_framework, method=''.join(mtd_str))
        case _:
            raise ValueError(f'wrong query type: {query_type}')
    return partial(search_mtd, trajs_info)
