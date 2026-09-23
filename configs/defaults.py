from yacs.config import CfgNode as CN


_C = CN()

_C.INDEX = CN()

_C.INDEX.REGION = CN()
# should the region index simply return all region or
# distinguish between regions surely in the query region from others
_C.INDEX.REGION.COARSE = False
_C.INDEX.REGION.TYPE = 'grid'

_C.INDEX.REGION.GRID = CN()
_C.INDEX.REGION.GRID.SPACE = (24, 20)

_C.INDEX.META_PATH = '/home/lg/VDBM/spatiotemporal/regional_tempo_spatial_query/meta'

_C.DATA = CN()
_C.DATA.PATH = '/media/lg/DataSet/Dataset/detection_results/'
_C.DATA.SCALE = 1
_C.DATA.STRIDE = 1
# Frames per second. Query `duration` and `interval` are authored in seconds and
# converted to frames (the unit used internally by the search and index layers)
# through this value. See utilities/time_unit.py.
_C.DATA.FPS = 30

_C.BOX_NP = True
