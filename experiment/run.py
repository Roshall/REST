import os.path
from functools import partial
from itertools import chain
from time import perf_counter as now

from configs import cfg
from ext_index.builder import load_traj_seg, build_rest
from scripts.border import load_ext_index_meta, border_stride
from search.sliding_based.framework import sliding_framework

from search.index_based.framework import index_based_framework
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


def init(query_type, trajs_info):
    query_type = query_type.split('_')
    match query_type:
        case ('index', *mtd_str):
            search_mtd = partial(index_based_framework, method=mtd_str)
        case ('sliding', *mtd_str):
            search_mtd = partial(sliding_framework, method=mtd_str)
        case _:
            raise ValueError(f'wrong query type: {cfg.QUERY}')
    return partial(search_mtd, trajs_info)


def query(mtd_str, trajs_info, content):
    search_mtd = init(mtd_str, trajs_info)
    return search_mtd(*content.values())


def count_result(mtd_str, searcher):
    count = 0
    start = now()
    for _ in searcher:
        count += 1
    end = now()
    print('-'.join(mtd_str), 'result count:', count, 'using', end - start, 's')


def find_bug(query_c, run):
    mtds_str = [('index', 'max_obj'), ('index', 'max_dur_multi')]
    searchers = [query(mtd_n, run, query_c) for mtd_n in mtds_str]
    res = [set((frozenset(ids), (s, e)) for ids, s, e in scher) for scher in searchers]
    for scher, r in zip(searchers, res):
        print(scher.__class__.__name__, len(r))
    print('1- 2 ', res[0] - res[1])
    print('2- 1 ', res[1] - res[0])
    return res


if __name__ == '__main__':
    dataset_name = 'ireland5h'
    interval_bound = 30 * 60
    query_content = {
        'region': Box2D((522,3794,237,1965), cfg),
        'pattern': {0: 1},  # label
        'duraiton': (1, 100000),
        'interval': (0, interval_bound),
    }
    file_path = os.path.join(cfg.DATA.PATH, f'{dataset_name}.pkl')
    # run the index based
    data_index = build_index(file_path)
    mtds_index = [f'index_{imtd}' for imtd in ('max_obj', 'max_dur_multi', 'max_dur_one', 'base')]
    # mtds_index = [f'index_{imtd}' for imtd in ('max_dur_multi', 'max_dur_one')]
    # run the sliding based
    # data_raw, _, _ = load_yolo_for(file_path)
    # mtds_sliding = [f'sliding_{smtd}' for smtd in ('naive', 'state')]
    mtds_sliding, data_raw = [], None
    for mtd, data in chain(((mtd, data_index) for mtd in mtds_index), ((mtd, data_raw) for mtd in mtds_sliding)):
        searcher = query(mtd, data, query_content)
        count_result(mtd, searcher)

    # res = find_bug(query_content, data_index)
