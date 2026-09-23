from yacs.config import CfgNode as CN


_C = CN()

_C.BBOX = (0, 1, 0, 1)
_C.PATTERN = {0: 2}
_C.DURATION = (5, 1000000)  # seconds
_C.INTERVAL = (0, 3 * 60 * 60)  # seconds
