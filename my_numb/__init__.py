import os

MY_NUMBA_ROOT = os.path.abspath(os.path.dirname(__file__))
os.environ['NUMBA_CACHE_DIR'] = os.path.join(MY_NUMBA_ROOT, 'code_cache')