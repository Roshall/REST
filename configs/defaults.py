from yacs.config import CfgNode as CN


_C = CN()

_C.INDEX = CN()
_C.INDEX.SEARCH_METHOD = CN()
_C.INDEX.SEARCH_METHOD.TEMPO = 'FuzzyInnerAll'
_C.INDEX.SEARCH_METHOD.REGION = 'NotSureSearch'
_C.INDEX.SEARCH_METHOD.USER = 'NotSureSearch'

_C.INDEX.REGION = CN()
# should the region index simply return all region or
# distinguish between regions surely in the query region from others
_C.INDEX.REGION.COARSE = False
_C.INDEX.REGION.TYPE = 'grid'

_C.INDEX.REGION.GRID = CN()
_C.INDEX.REGION.GRID.SPACE = (16, 12)

_C.INDEX.CONFIG_PATH = '/home/lg/VDBM/spatiotemporal/regional_tempo_spatial_query/meta'

_C.DATA = CN()
_C.DATA.PATH = '/home/lg/VDBM/spatiotemporal/regional_tempo_spatial_query/test'
_C.DATA.SCALE = 1
_C.DATA.STRIDE = 1

_C.BOX_NP = True

# _C.QUERY = ('index', 'one_pass')
# _C.QUERY = ('index', 'base')
# _C.QUERY = ('sliding', 'base')
# _C.QUERY = ('sliding', 'state')
