from functools import partial

from btree import BtreeMap, BtreeMultiMap

from indices.region import GridRegion, unsure
from test.region_test_vars import *
from indices.two_level_merge_index import TwoLevelMergeIndex


class TestIndices:
    Out3D = partial(TwoLevelMergeIndex, GridRegion, BtreeMap)
    Tempo2D = partial(TwoLevelMergeIndex, BtreeMap, BtreeMultiMap)
    Tempo2D_fuzzy = partial(TwoLevelMergeIndex, BtreeMap, BtreeMultiMap, 'FuzzySearch')

    def test_out3d(self):
        out3d = self.Out3D()
        for key, obj, in zip(keys, objs):
            out3d.add(key, obj)
            assert out3d.where_contain(key) is obj

        res = list(out3d.where_intersect([(20, 30, 16, 70), (10, 100)]))
        assert len(res) == 4

    def test_tempo2D(self):
        tempo2d = self.Tempo2D()

        for *key, entry in zip(durations, begin, content):
            tempo2d.add(key, entry)
            assert tempo2d.where_contain(key) == entry

        res = list(tempo2d.where_intersect([(2,100),(1,100)]))
        assert len(res) == 4

    def test_user_idx(self):
        user_idx = TwoLevelMergeIndex(self.Out3D, self.Tempo2D)
        for *key, entry in zip(keys, zip(durations, begin), content):
            user_idx.add(key, entry)
            assert user_idx.where_contain(key) == entry
        # search
        res = list(user_idx.where_intersect([((15, 30, 16, 70), (10, 110)), ((2, 100), (1, 100))]))
        assert len(res) == 3

    def test_fuzzy(self):
        fuzzy_user = TwoLevelMergeIndex(self.Out3D, self.Tempo2D_fuzzy)
        for *key, entry in zip(keys, zip(durations, begin), content):
            fuzzy_user.add(key, entry)
        res = list(fuzzy_user.where_intersect([((15, 30, 16, 70), (10, 110)), (46, 67)]))
        assert len(res) == 3

    def test_fuzzy_inner_all(self):
        fuzzy_tempo_inner_all = partial(TwoLevelMergeIndex, BtreeMap, BtreeMultiMap, 'FuzzyInnerAll')
        user_idx = TwoLevelMergeIndex(self.Out3D, fuzzy_tempo_inner_all)
        for *key, entry in zip(keys, zip(durations, begin), content):
            user_idx.add(key, entry)
        tempo_set = user_idx.where_intersect([((15, 30, 16, 70), (10, 110)), (46, 67)])
        res = [entry for temp in tempo_set for entry in temp]
        assert len(res) == 3

    def test_perhaps_intersect(self):
        out3D = partial(TwoLevelMergeIndex, unsure(GridRegion), BtreeMap, 'NotSureSearch')
        user_perhaps = TwoLevelMergeIndex(out3D, self.Tempo2D_fuzzy, 'NotSureSearch')
        for *key, entry in zip(keys, zip(durations, begin), content):
            user_perhaps.add(key, entry)
        candidates, probation = user_perhaps.where_intersect([((15, 30, 16, 70), (10, 110)), (46, 67)])
        assert len(list(candidates)) == 0
        assert len(list(probation)) == 3
