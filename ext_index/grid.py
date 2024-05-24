import numpy as np


class Grid:
    def __init__(self, border_stride):
        border_stride = np.array(border_stride)
        self.lower_border = np.array(border_stride[[0, 3]])
        self.stride = np.array(border_stride[[2, 5]])
        box = border_stride[[0, 1, 3, 4]].reshape(-1, 2)
        box = (box[:, 1] - box[:, 0]) // np.asarray(self.stride) + 1
        self.size = box[0] * box[1]
        self.col_num = box[0]

    def index(self, point):
        pos = (point - self.lower_border)//self.stride
        pos = pos[:, 1] * self.col_num + pos[:, 0]
        if (pos >= self.size).any():
            raise ValueError(f"points outside grid")
        return pos
