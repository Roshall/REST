import os.path
from functools import partial
from time import perf_counter as now

from configs import cfg
from ext_index.builder import load_traj_seg, build_rest
from scripts.border import load_ext_index_meta, border_stride
from search.base import base_search
from search.baseline.search_methods import sliding_framework
from search.max_dur_first import max_dur_first_search
from search.max_obj import max_obj_search
from utilities.box2D import Box2D


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


def init(query_type, data):
    match query_type:
        case ('index', mtd):
            # frame_info[0].query("fid <= 72000", inplace=True)
            match mtd:
                case 'max_dur':
                    search_mtd = max_dur_first_search
                case 'base':
                    search_mtd = base_search
                case 'max_obj':
                    search_mtd = max_obj_search
                case _:
                    raise ValueError(f'No index based {mtd} found')
        case ('sliding', mtd):
            search_mtd = partial(sliding_framework, method=mtd)
        case _:
            raise ValueError(f'wrong query type: {cfg.QUERY}')
    return partial(search_mtd, data)


def query(mtd_str, data):
    interval_bound = 30 * 60 * 60 * 3
    query_content = [
        Box2D((577, 1537, 444, 984), cfg),  # region
        {0: 1},  # label
        (5, 10000),  # duration
        (0, interval_bound),  # interval
    ]
    search_mtd = init(mtd_str, data)
    return search_mtd(*query_content)


def count_result(mtd, searcher):
    count = 0
    start = now()
    for _ in searcher:
        count += 1
    end = now()
    print('-'.join(mtd), 'result count:', count, 'using', end - start, 's')


def find_bug():
    mtds = [('index', 'max_obj'), ('index', 'max_dur')]
    searchers = [query(mtd, data) for mtd in mtds]
    res = [set((frozenset(ids), (s, e)) for ids, s, e in scher) for scher in searchers]
    for scher, r in zip(searchers, res):
        print(scher.__class__.__name__, len(r))
    print('1- 2 ', res[0] - res[1])
    print('2- 1 ', res[1] - res[0])
    return res



if __name__ == '__main__':
    dataset_name = 'shinjuku3h'
    file_path = os.path.join(cfg.DATA.PATH, f'{dataset_name}.pkl')
    data = build_index(file_path)
    mtds = [('index', 'max_obj')]
    for mtd in mtds:
        searcher = query(mtd, data)
        count_result(mtd, searcher)
    # res = find_bug()
