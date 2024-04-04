from bisect import bisect_right
import numpy as np

from utilities.data_preprocessing import gen_border


class GridRegion:
    def __init__(self, cfg):
        self.borders = gen_border(cfg.INDEX.REGION.GRID.AREA, cfg.INDEX.REGION.GRID.SPACE)
        # this marker marks the starting point of each grid, while $borders
        # contains the end corners, thus removing the last element
        self.terr_marker = [dim[:-1] for dim in self.borders]
        shape = tuple(map(len, self.terr_marker))
        self.territory = np.empty(shape[::-1], dtype=object)  # x: column, y: row etc.

    def add(self, loc, entry):
        self.territory[self._point2idx(loc)] = entry

    def _point2idx(self, loc):
        return tuple(
            reversed([bisect_right(dim_grid, dim_point) - 1 for dim_grid, dim_point in zip(self.terr_marker, loc)]))

    def _enclose(self, loc):
        for dim_grid, dim_point in zip(self.borders, loc):
            if dim_point < dim_grid[0] or dim_point > dim_grid[-1]:
                return False
        return True

    def where_contain(self, loc):
        if not self._enclose(loc):
            return
        return self.territory[self._point2idx(loc)]

    def index(self, locs):
        # check locs borders ####
        # loc_t = locs.T
        # mask = np.ones(locs.shape[0], dtype=bool)
        # for dim_grid, dim_point in zip(self.borders, loc_t):
        #     mask &= (dim_point >= dim_grid[0]) | (dim_point <= dim_grid[-1])
        # assert mask.all()
        # ################
        res = np.zeros(locs.shape[0], dtype=np.int32)
        for i, (dim_grid, dim_point) in enumerate(zip(self.borders, locs.T)):
            res += (np.searchsorted(dim_grid, dim_point, side='right') - 1) * (len(dim_grid) if i < 1 else 1)
        return res

    def where_intersect(self, bbox):
        bbox_iter = iter(bbox)
        region_ls = []
        for dim_grid in self.borders:
            dmin = bisect_right(dim_grid, next(bbox_iter)) - 1
            dmax = bisect_right(dim_grid, next(bbox_iter), dmin)  # the end of a slice is exclusive
            region_ls.append(slice(dmin, dmax))
        res = self.territory[tuple(reversed(region_ls))]
        return res[res.nonzero()]

    def perhaps_intersect(self, bbox):
        bbox_iter = iter(bbox)
        probation_slice = []
        candidate_slice = []
        for dim_grid in self.borders:
            border_min = next(bbox_iter)
            cand_min = np.searchsorted(dim_grid, border_min, side='right') - 1
            if dim_grid[cand_min] == border_min:  # is the border grid surely within the bbox?
                prob_min = cand_min
            else:
                prob_min = cand_min + 1

            border_max = next(bbox_iter)
            if border_max >= dim_grid[-1]:
                prob_max = len(dim_grid)  # greater than territory's len is ok
                cand_max = prob_max
            else:
                cand_max = bisect_right(dim_grid, border_max, cand_min)  # the end of a slice is exclusive
                prob_max = cand_max if border_max == dim_grid[cand_max] - 1 else cand_max - 1
            probation_slice.append(slice(prob_min, prob_max))
            candidate_slice.append(slice(cand_min, cand_max))
        probation = self.territory[tuple(reversed(probation_slice))]
        probation = probation[probation.nonzero()]
        candidate = self.territory[tuple(reversed(candidate_slice))]
        candidate = candidate[candidate.nonzero()]
        return np.setdiff1d(candidate, probation, assume_unique=True), probation


class Out3DRegion:
    def __init__(self, data=None):
        import btree
        self.duration_index = btree.BtreeMap()
        if data is not None:
            for key, entry in data:
                self.add(key, entry)

    def where_contain(self, key):
        return (reg := self.duration_index.where_contain(key.duration)) and reg.where_contain(key.loc)

    def where_intersect(self, bbox):
        for reg in self.duration_index.where_intersect(bbox[:2]):
            yield from reg.where_intersect(bbox[2:])

    def add(self, key, entry) -> None:
        if (reg := self.duration_index.where_contain(key.duration)) is None:
            reg = GridRegion()
            self.duration_index.add(key.duration, reg)
        reg.add(key.loc, entry)


def unsure(cls):
    def unsure_cls():
        instance = cls()
        instance.where_intersect = instance.perhaps_intersect
        return instance
    return unsure_cls


class TrajectoryClusteringRegion:
    def add(self, key, entry) -> None:
        pass

    def __init__(self, data):
        pass

    def where_contain(self, loc):
        pass

    def where_intersect(self, bbox):
        pass
