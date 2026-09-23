from functools import partial

from search.index_based.framework import index_based_framework
from search.sliding_based.framework import sliding_framework


def query_init(query_type, trajs_info, fps=None):
    """Bind a search method. ``duration``/``interval`` are seconds unless the
    bound method is given ``fps=1``, which means frames."""
    query_type = query_type.split('_')
    match query_type:
        case ('index', *mtd_str):
            search_mtd = partial(index_based_framework, method=''.join(mtd_str), fps=fps)
        case ('sliding', *mtd_str):
            search_mtd = partial(sliding_framework, method=''.join(mtd_str), fps=fps)
        case _:
            raise ValueError(f'wrong query type: {query_type}')
    return partial(search_mtd, trajs_info)
