from functools import partial

import btree

from .region import GridRegion, unsure
from .two_level_merge_index import TwoLevelMergeIndex

# Here we need duration to deduct the correct beginning of a segment
# overlapping within the searching interval. Therefore, fuzzy search is needed.
# key: (duration, begin)
# Tempo2DIdx = partial(TwoLevelMergeIndex, btree.BtreeMap, btree.BtreeMultiMap, 'FuzzyInnerAll')
# key: ((x, y), duration)
# Out3DRegion = partial(TwoLevelMergeIndex, unsure(GridRegion), btree.BtreeMap, 'NotSureSearch')
# key:(((x, y), duration), (duration, begin))
# UserIdx = partial(TwoLevelMergeIndex, Out3DRegion, Tempo2DIdx, 'NotSureSearch')


def get_user_indices(cfg):
    Tempo2DIdx = partial(TwoLevelMergeIndex, btree.BtreeMap, btree.BtreeMultiMap, cfg.INDEX.SEARCH_METHOD.TEMPO)
    grid_region = partial(GridRegion, cfg)
    Out3DRegion = partial(TwoLevelMergeIndex, grid_region if cfg.INDEX.REGION.COARSE else unsure(grid_region),
                          btree.BtreeMap, cfg.INDEX.SEARCH_METHOD.REGION)
    return partial(TwoLevelMergeIndex, Out3DRegion, Tempo2DIdx, cfg.INDEX.SEARCH_METHOD.USER)
