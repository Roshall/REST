from indices import build_tempo_spatial_index
from utilities import dataset
from utilities.box2D import Box2D
from utilities.config import config
from utilities.data_preprocessing import traj_data, gen_border, view_field


def grid_spt_tempo_idx_fake_data(cfg):
    data, cols, cls_map, = dataset.load_fake()
    trajs = traj_data(data, cols, cls_map, 1, 1)
    broders = gen_border(view_field(data[cols[-2:]]), (5, 6))
    config.gird_border = broders
    return build_tempo_spatial_index(trajs, cfg)


def query4test():
    test1 = [Box2D((0, 400, 0, 500)),  # region
             {0: 1},  # label
             (2, 30),  # duration
             (0, 10)]  # interval

    test2 = [Box2D((100, 300, 100, 400)),
             {0: 2, 1: 1},
             (10, 30),
             (0, 50)]
    return test1, test2
