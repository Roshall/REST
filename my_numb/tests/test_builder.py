import os.path

import numpy as np
import pytest

from my_numb.search.index_based.builder import save_index_meta, partition_traj, load_segments, cppyy_init, \
    flatten_border
from my_numb.utilities.data_processing import traj_df2np
from scripts.border import concatenate_stride
from utilities.dataset import load_fake


@pytest.fixture
def package():
    return load_fake(False)

@pytest.fixture
def traj_data(package):
    return traj_df2np(package[0], package[1])

def test_trja2np(traj_data):
    ids, labels, begins, offsets, points = traj_data
    assert np.array_equal(ids, np.arange(1, 6))
    assert np.array_equal(labels, np.repeat([0, 1], [3, 2]))
    assert np.array_equal(begins, np.array([1, 5, 4, 2, 5]))
    assert np.array_equal(offsets, np.array([0, 29, 57, 83, 111, 136]))
    assert len(points) == 136 and points.shape == (136, 2)



def test_save_index_meta(traj_data):
    fields = ['ids', 'cls', 'begins', 'offsets', 'points']

    save_index_meta(traj_data, fields, 'test_meta')
    os.path.exists('test_meta.npz')
    trajs_dict = np.load('test_meta.npz')
    assert np.array_equal(trajs_dict['ids'], np.arange(1, 6))
    assert np.array_equal(trajs_dict['cls'], np.repeat([0, 1], [3, 2]))
    assert np.array_equal(trajs_dict['begins'], np.array([1, 5, 4, 2, 5]))
    assert np.array_equal(trajs_dict['offsets'], np.array([0, 29, 57, 83, 111, 136]))
    assert len(trajs_dict['points']) == 136 and trajs_dict['points'].shape == (136, 2)

@pytest.fixture
def border_m():
    border_m = {}
    reg = [0, 400, 0, 500]
    life_border = [3, 8, 10, 21]
    for c in (0, 1):
        border_m[c] = {'reg_border': reg, 'tempo_stride': 6, 'life_border': life_border}
    return border_m

@pytest.fixture
def space():
    return 4, 5

def test_partition_traj(traj_data, border_m, space):
    bs_m = {}
    for c, bs in concatenate_stride(border_m, space).items():
        bs_m[c] = np.array(bs)
    segs, offsets = partition_traj(traj_data, bs_m)
    assert len(offsets) == 6
    assert np.all(segs[offsets[:-1]] == 0)

def test_load_segments(traj_data, border_m, space):
    segs, offsets = load_segments(traj_data, 'test', border_m, '.', space)
    seg_meta = np.load('test(4, 5).npz')
    assert np.array_equal(segs, seg_meta['segs'])
    assert np.array_equal(offsets, seg_meta['offsets'])

def test_border_init(border_m, space):
    import cppyy
    from my_numb.cxx_store import LIB_DIR, LIB_NAME, HEADER
    cppyy.add_include_path(os.path.join(LIB_DIR, 'include'))
    cppyy.include(HEADER)
    cppyy.add_library_path(os.path.join(LIB_DIR, 'lib'))
    cppyy.load_library(LIB_NAME)
    concatenate_stride(border_m, space)
    # cppyy_init(cppyy)
    borders = []
    for c, border in border_m.items():
        flatten_border(c, borders, border)
    cppyy.gbl.border_init(borders)
    reg_0 = list(cppyy.gbl.get_border(0, 0))
    assert reg_0 == border_m[0]['border_stride']
    assert list(cppyy.gbl.get_border(0, 1)) == border_m[0]['life_border']
    assert list(cppyy.gbl.get_border(1, 0)) == border_m[1]['border_stride']
    assert list(cppyy.gbl.get_border(1, 1)) == border_m[1]['life_border']