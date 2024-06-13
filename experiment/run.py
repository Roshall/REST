import os.path
from functools import partial
from itertools import chain
from time import perf_counter as now

from configs import cfg
from ext_index.builder import load_traj_seg, build_rest
from scripts.border import load_ext_index_meta, border_stride
from search.base import base_search
from search.baseline.search_methods import sliding_framework
from search.max_dur_first import max_dur_first_search
from search.max_obj import max_obj_search
from utilities.box2D import Box2D
from utilities.dataset import load_yolo_for


def build_index(fname):
    bor_m = load_ext_index_meta(os.path.join(cfg.INDEX.META_PATH, f'{dataset_name}.json'))
    bs_m = {}
    for c, border in bor_m.items():
        bs = border_stride(border['reg_border'], cfg.INDEX.REGION.GRID.SPACE)
        bs_m[c] = bs
        border['border_stride'] = bs
    trajs, segs = load_traj_seg(bs_m, fname, dataset_name, cfg)
    spt_idx = build_rest(trajs, segs, bor_m)
    return spt_idx, trajs


def init(query_type, trajs_info):
    match query_type:
        case ('index', mtd_str):
            match mtd_str:
                case 'max_dur':
                    search_mtd = max_dur_first_search
                case 'base':
                    search_mtd = base_search
                case 'max_obj':
                    search_mtd = max_obj_search
                case _:
                    raise ValueError(f'No index based {mtd_str} found')
        case ('sliding', mtd_str):
            search_mtd = partial(sliding_framework, method=mtd_str)
        case _:
            raise ValueError(f'wrong query type: {cfg.QUERY}')
    return partial(search_mtd, trajs_info)


def query(mtd_str, trajs_info, content):
    search_mtd = init(mtd_str, trajs_info)
    return search_mtd(*content)


def count_result(mtd_str, searcher):
    count = 0
    start = now()
    for _ in searcher:
        count += 1
    end = now()
    print('-'.join(mtd_str), 'result count:', count, 'using', end - start, 's')


def find_bug(query_c):
    mtds_str = [('index', 'max_obj'), ('index', 'max_dur')]
    searchers = [query(mtd_n, data, query_c) for mtd_n in mtds_str]
    res = [set((frozenset(ids), (s, e)) for ids, s, e in scher) for scher in searchers]
    for scher, r in zip(searchers, res):
        print(scher.__class__.__name__, len(r))
    print('1- 2 ', res[0] - res[1])
    print('2- 1 ', res[1] - res[0])
    return res


if __name__ == '__main__':
    dataset_name = 'florida5h'
    interval_bound = 30 * 60 * 60 * 3
    query_content = [
        Box2D((250,1786,38,902), cfg),  # region
        {0: 2},  # label
        (10, 10000),  # duration
        (0, interval_bound),  # interval
    ]
    file_path = os.path.join(cfg.DATA.PATH, f'{dataset_name}.pkl')
    # run the index based
    data_index = build_index(file_path)
    # mtds_index = [('index', imtd) for imtd in ('max_obj', 'max_dur', 'base')]
    mtds_index = [('index', imtd) for imtd in ('max_obj', 'max_dur')]
    # run the sliding based
    # data_raw, _, _ = load_yolo_for(file_path)
    # mtds_sliding = [('sliding', smtd) for smtd in ('naive', 'state')]
    mtds_sliding, data_raw = [], None
    for mtd, data in chain(((mtd, data_index) for mtd in mtds_index), ((mtd, data_raw) for mtd in mtds_sliding)):
        searcher = query(mtd, data, query_content)
        count_result(mtd, searcher)

    # res = find_bug(query_content)
