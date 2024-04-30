import os.path
import pickle
from functools import partial
from time import perf_counter as now

from configs import cfg
from indices.builder import build_tempo_spatial_index, build_index_from_segments, save_trajectory_segments
from search.base import base_search
from search.baseline.search_methods import sliding_framework
from search.one_pass import one_pass_search
from utilities.box2D import Box2D
from utilities.data_preprocessing import traj_data, view_field
from utilities.dataset import load_yolo_for

wd = os.path.dirname(os.path.abspath(__file__))


def get_index(frame_info, cfg):
    trajs = traj_data(*frame_info, cfg.DATA.STRIDE, scale=cfg.DATA.SCALE)
    temp_spt = build_tempo_spatial_index(trajs, cfg)
    return temp_spt


def init(query_type, frame_info):
    match query_type:
        case ('index', mtd):
            # frame_info[0].query("fid <= 72000", inplace=True)
            match mtd:
                case 'one_pass':
                    search_mtd = one_pass_search
                case 'base':
                    cfg.merge_from_file(os.path.join(wd, 'configs', 'region_base.yml'))
                    search_mtd = base_search
                case _:
                    raise ValueError(f'No index based {mtd} found')
            # fov = view_field(frame_info[0][['x', 'y']])
            # cfg.merge_from_list(['INDEX.REGION.GRID.AREA', tuple(fov)])
            seg_path = '/home/lg/VDBM/spatiotemporal/resource/segments.pkl'
            if os.path.exists(seg_path):
                with open(seg_path, 'rb') as f:
                    segments = pickle.load(f)
            else:
                trajs = traj_data(*frame_info, cfg.DATA.STRIDE, scale=cfg.DATA.SCALE)
                segments = save_trajectory_segments(trajs, cfg, seg_path)
            data = build_index_from_segments(segments, cfg)
        case ('sliding', mtd):
            data = frame_info[0]
            search_mtd = partial(sliding_framework, method=mtd)
        case _:
            raise ValueError(f'wrong query type: {cfg.QUERY}')
    return partial(search_mtd, data)


def query():
    mtd_str = ('index', 'one_pass')
    interval_bound = 60*60*20
    query_content = [
        Box2D((2775, 3472, 746, 1137), cfg),  # region
        {0: 2},  # label
        (5, 10000),  # duration
        (0, interval_bound),  # interval
    ]
    search_mtd = init(mtd_str, load_yolo_for(filename))
    count = 0
    start = now()
    for _ in search_mtd(*query_content):
        count += 1
    end = now()
    print('_'.join(mtd_str), 'result count:', count, 'using', end - start, 's')


def find_bug():
    interval_bound = 60*60*30
    query_content = [
        Box2D((2775, 3472, 746, 1137), cfg),  # region
        {0: 2},  # label
        (5, 10000),  # duration
        (0, interval_bound),  # interval
    ]
    mtds = [('index', 'one_pass'), ('index', 'base'), ('sliding', 'base'), ('sliding', 'state')]
    res = [set((frozenset(ids), (s, e)) for ids, s, e in init(mtd_str, load_yolo_for(filename))(*query_content)) for mtd_str in mtds]
    # canary_mtd = init(('index', 'base'))
    # one = set((frozenset(ids), (s, e)) for ids, s, e in canary_mtd(*query_content))
    # test_mtd = init(('sliding', 'state'))
    # two = set((frozenset(ids), (s, e)) for ids, s, e in test_mtd(*query_content))
    with open('find_bug.pkl', 'wb') as f:
        import pickle
        pickle.dump(res, f)
    # for i, ms, re in enumerate(zip(mtds, res)):
    #     print('_'.join(ms), 'num:', len(res), 'result:', re)
    #     if i != 0:
    #         print(f"in {'_'.join(mtds[0])} but not in {'_'.join(ms)}", res[0] - re)
    #         print(f"in {'_'.join(ms)} but not in {'_'.join(mtds[0])}", re - res[0])

    # print('canary_res_num:', len(one))
    # print('one_pass_res:', one)
    # print('test_res_num:', len(two))
    # print('In canary but not test: ', one - two)
    # print('In test but not canary: ', two - one)
    #


if __name__ == '__main__':
    filename = '/media/lg/DataSet/Dataset/detection_results/timsquare3h.pkl'
    # query()
    find_bug()
