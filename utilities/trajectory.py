from dataclasses import dataclass
from typing import Sequence, NamedTuple

import numpy as np


class TrajTrak(NamedTuple):
    tId: int
    start_frame: int
    clsId: int
    track: np.ndarray


class RawTraj(NamedTuple):
    fps: int
    life_long: int
    bbox: Sequence
    traj_track: TrajTrak


class TrajMeta(NamedTuple):
    """ This class should be removed"""
    duration: Sequence
    loc: int


@dataclass(slots=True)
class Trajectory:
    id: int
    label: int
    seg: list[int]


@dataclass(slots=True)
class BasicTrajectorySeg:
    id: int
    begin: int
    label: int

    def __lt__(self, other):
        return self.begin < other.begin

    def __eq__(self, other):
        return self.begin == other.begin


@dataclass(slots=True)
class TrajectoryIntervalSeg(BasicTrajectorySeg):
    len: int


@dataclass(slots=True)
class TrajectorySequenceSeg(BasicTrajectorySeg):
    points: Sequence

    @property
    def len(self):
        return len(self.points)
