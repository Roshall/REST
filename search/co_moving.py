from collections import Counter
from collections.abc import Mapping, MutableSequence
from dataclasses import dataclass


@dataclass(slots=True)
class CoMovementPattern:
    labels: Mapping  # dict of id -> label
    interval: MutableSequence = None

    @property
    def objs(self):
        return self.labels.keys()

    @property
    def start(self):
        return self.interval[0]

    @start.setter
    def start(self, val: int):
        self.interval[0] = val

    @property
    def end(self):
        return self.interval[1]

    @end.setter
    def end(self, val: int):
        self.interval[1] = val

    def label_count(self):
        return Counter(self.labels.values())

    def to_plain(self):
        return tuple(self.labels), *self.interval

    def __len__(self):
        return len(self.labels)

    def __and__(self, other):
        intersect_objs = self.objs & other.objs
        cls = type(self)
        return cls({obj: self.labels[obj] for obj in intersect_objs})

    def __eq__(self, other):
        return tuple(self.interval) == tuple(other.interval) and self.objs == other.objs

    def __str__(self):
        return str(self.to_plain())
