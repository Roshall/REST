from functools import partial

import numpy as np


class Box:
    def __init__(self, bbox):
        """
        build box by bbox
        :param bbox: (dim0_min, dim0_max, dim1_min, dim1_max)
        """
        self.bbox = bbox
        self._test_meta = None
        self._meta()

    def _point_rep(self):
        """
        transform bbox into 3 points to represent a box
        p[1](A)
        |
        |
        p[0](B)____p[2](C)
        :return: three points
        """
        points = np.empty((3, 2))
        points[0] = self.bbox[::2]
        points[1] = self.bbox[::3]
        points[2] = self.bbox[1:3]
        return points

    def _meta(self):
        """
        build useful data for testing if a point within this box
        :return: None
        """
        points_rep = self._point_rep()
        reference = np.empty((4, len(points_rep[0])), dtype=np.uint32)
        vectors = points_rep[1:] - points_rep[0]  # [BA, BC].T
        reference[:2] = vectors.T
        reference[2] = np.diag(vectors @ vectors.T)  # [BA@BA, BC@BC]
        reference[3] = points_rep[0]
        self._test_meta = reference

    def enclose(self, points):
        """
        test whether points are in this box
        :param points: 2d points, iterable
        :return: boolean array
        """
        stats = points - self._test_meta[-1]  # BP
        stats = stats @ self._test_meta[:2]  # [BP@BA.T, BP@BC.T]
        mask = stats >= 0
        mask &= stats <= self._test_meta[2]  # [BP@BA < BA@BA, BP@BC < BC@BC]
        return mask.all(axis=1)


class Box2D:
    def __init__(self, bbox, config=None):
        """
        build box by bbox
        :param bbox: (dim0_min, dim0_max, dim1_min, dim1_max)
        """
        self.bbox = bbox
        self._test_meta = None
        self._meta()
        us_np = True if config is None else config.BOX_NP
        self.enclose = self.enclose_parallel if us_np else self.enclose_serial

    def rest_bbox(self, bbox):
        self.bbox = bbox
        self._meta()

    def _meta(self):
        """
        build useful data for testing if a point within this box
        :return: None
        """
        bbox = np.array(self.bbox).reshape(-1, 2).T
        bbox_np = np.ascontiguousarray(bbox, dtype=np.uint16)
        self._test_meta = bbox_np
        self.verifier = partial(region_verify, bbox_np)

    def enclose_parallel(self, points):
        """
        test whether points are in this box
        :param points: 2d points, iterable
        :return: boolean array
        """
        return np.logical_and.reduce((points >= self._test_meta[0]) & (points <= self._test_meta[1]), axis=1)

    def enclose_serial(self, points):
        return np.apply_along_axis(self.verifier, 1, points)


def region_verify(bound, points):
    res = ((points >= bound[0]) & (points <= bound[1])).all()
    return res
