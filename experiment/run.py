import os.path
from time import perf_counter as now

from configs import cfg
from utilities.box2D import Box2D
from search_cxx.cxx_search import CxxIndex, cxx_sliding_query


def count_result(mtd_str, searcher):
    count = 0
    start = now()
    for _ in searcher:
        count += 1
    end = now()
    print('-'.join(mtd_str), 'result count:', count, 'using', end - start, 's')


if __name__ == '__main__':
    # The Python experiment end now runs entirely on the C++ search engine
    # (search_cxx.cxx_search). The legacy Python index (traj_seg/index_w) was
    # removed, so we build the index with CxxIndex and query through it (and
    # through cxx_sliding_query for the sliding-based methods).
    dataset_name = 'ireland5h'
    interval_bound = 30 * 60  # seconds: 30 minutes
    query_content = {
        'region': Box2D((522,3794,237,1965), cfg),
        'pattern': {0: 1},  # label
        'duraiton': (1, 100000),  # seconds
        'interval': (0, interval_bound),  # seconds
    }
    parquet_path = os.path.join(cfg.DATA.PATH, f'{dataset_name}.parquet')
    meta_path = os.path.join(cfg.INDEX.META_PATH, f'{dataset_name}.json')
    cxx_idx = CxxIndex(parquet_path, meta_path,
                       cfg.INDEX.REGION.GRID.SPACE[0],
                       cfg.INDEX.REGION.GRID.SPACE[1],
                       cfg.DATA.STRIDE, cfg.DATA.SCALE)
    bbox = query_content['region'].bbox
    pattern = query_content['pattern']
    dur = query_content['duraiton'][0]
    interval = query_content['interval']

    for imtd in ('max_obj', 'max_dur_multi', 'max_dur_one'):
        count_result(('index', imtd),
                     cxx_idx.query(bbox, pattern, dur, interval, imtd))
    for smtd in ('naive', 'state'):
        count_result(('sliding', smtd),
                     cxx_sliding_query(parquet_path, bbox, pattern, dur,
                                       interval, smtd))
