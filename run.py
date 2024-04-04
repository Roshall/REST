from functools import partial
from time import perf_counter as now

from configs import cfg
from indices.builder import build_tempo_spatial_index
from search.base import base_search
from search.baseline.search_methods import sliding_framework
from search.one_pass import one_pass_search
from utilities.box2D import Box2D
from utilities.data_preprocessing import traj_data
from utilities.dataset import load_yolo_for


def get_index(frame_info, cfg):
    trajs = traj_data(*frame_info, cfg.DATA.STRIDE, scale=cfg.DATA.SCALE)
    # broders = gen_border(trajs.bbox, 16, 11)
    temp_spt = build_tempo_spatial_index(trajs, cfg)
    return temp_spt


def init(query_type):
    match query_type:
        case ('index', mtd):
            frame_info = load_yolo_for(filename)
            frame_info[0].query("3600 <= fid <= 18000", inplace=True)
            data = get_index(frame_info, cfg)
            match mtd:
                case 'one_pass':
                    search_mtd = one_pass_search
                case 'base':
                    cfg.merge_from_file('configs/region_base.yml')
                    search_mtd = base_search
                case _:
                    raise ValueError(f'No index based {mtd} found')
        case ('sliding', mtd):
            data, _, _ = load_yolo_for(filename)
            search_mtd = partial(sliding_framework, method=mtd)
        case _:
            raise ValueError(f'wrong query type: {cfg.QUERY}')
    return partial(search_mtd, data)


def query():
    interval_bound = 60*60*70
    query_content = [
        Box2D((3280, 3551, 1429, 1665)),  # region
        {0: 1},  # label
        (5, 100),  # duration
        (0, interval_bound),  # interval
    ]
    # query_content[0] = Box2D((700, 1000, 427, 569))
    search_mtd = init(('index', 'one_pass'))
    count = 0
    start = now()
    for _ in search_mtd(*query_content):
        count += 1
    end = now()
    print(cfg.QUERY, 'result count:', count, 'using', end - start, 's')


def find_bug():
    interval_bound = 60*60*70
    query_content = [
        Box2D((3280, 3551, 1429, 1665)),  # region
        {0: 1},  # label
        (5, 100),  # duration
        (60*60, interval_bound),  # interval
    ]
    query_content[0] = Box2D((700, 1000, 427, 569))

    canary_mtd = init(('index', 'one_pass'))
    one = set((frozenset(ids), (s, e)) for ids, s, e in canary_mtd(*query_content))
    test_mtd = init(('sliding', 'base'))
    two = set((frozenset(ids), (s, e)) for ids, s, e in test_mtd(*query_content))

    print('canary_res_num:', len(one))
    print('one_pass_res:', one)
    print('test_res_num:', len(two))
    print('In canary but not test: ', one - two)
    print('In test but not canary: ', two - one)


filename = '../resource/dataset/traj_taipei_0412.pkl'
# query()
find_bug()
