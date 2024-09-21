import copy
from collections.abc import Mapping
from itertools import groupby, islice

from configs import cfg
from ext_index.builder import load_seg, build_rest
from profile import Profile
from utilities.query import build_index, query_init
from helper import *
from scripts.border import load_ext_index_meta
from utilities.dataset import load_yolo_for


def interval_exp(default_query: dict, interval_meta: Mapping,
                 ds_name, clip_long='3h'):
    query_cand = copy.deepcopy(default_query)
    if clip_long not in ('3h', '5h'):
        raise ValueError('lf must be either 3h or 5h')
    interval_end_ls = interval_meta[clip_long]
    for end in interval_end_ls:
        query_cand['interval'][-1] = end * 30 * 60
        yield query_cand


def duration_exp(default_query, dur_meta, ds_name):
    query_cand = copy.deepcopy(default_query)
    if not isinstance(dur_meta, list) or len(dur_meta) != 3:
        raise ValueError('dur_meta must be a list of length 3')
    print('dur:', end=' ')
    for d in range(*dur_meta):
        print(d, end=' ')
        query_cand['duration'][0] = d
        yield query_cand


def region_exp(default_query: dict, region_meta: Mapping, ds_name):
    query_cand = copy.deepcopy(default_query)
    for bboxes in region_meta['bboxes'][ds_name]:
        query_cand['region'] = bboxes
        yield query_cand


def obj_num_exp(default_query: dict, obj_meta, ds_name):
    query_cand = copy.deepcopy(default_query)
    if not isinstance(obj_meta, list) or len(obj_meta) != 3:
        raise ValueError('obj_meta must be a list of length 3')
    pattern = query_cand['pattern']
    for n in range(*obj_meta):
        for obj in pattern:
            pattern[obj] = n
        yield query_cand


def cat_num_exp(default_query: dict, cat_meta, ds_name):
    query_cand = copy.deepcopy(default_query)
    if not isinstance(cat_meta, list):
        raise ValueError('cat_meta must be a list')
    cat_ls = list(range(*cat_meta))
    print('cat:', end=' ')
    for idx in range(1, len(cat_ls) + 1):
        print(cat_ls[:idx], end=' ')
        query_cand['pattern'] = {obj: 1 for obj in cat_ls[:idx]}
        yield query_cand

def default_exp(default_query: dict, region_meta: Mapping, ds_name):
    yield default_query


def mod_grid(grid_meta: Mapping):
    """
    rebuild index after fetch one grid
    """
    if not isinstance(grid_meta, list) or len(grid_meta) != 2:
        raise ValueError('grid_meta must be a list of length 2')
    for w in range(*grid_meta[0]):
        for h in range(*grid_meta[1]):
            yield w, h


def argument_region(o_plans, rnum):
    for p in o_plans:
        group_p = []
        regs = p['region']
        for reg in islice(regs, rnum):
            p_c = copy.deepcopy(p)
            reset_region(p_c, reg, cfg)
            group_p.append(p_c)
        yield group_p


func_m = {'region': region_exp, 'obj_num': obj_num_exp, 'cat_num': cat_num_exp,
          'interval': interval_exp, 'duration': duration_exp,
          'default': default_exp}


def run_one_exp(search_mtd, plans, region_num, repeat):
    for plan_g in argument_region(plans, region_num):
        pos = Profile()
        results = []
        count = 0
        for i, plan in enumerate(plan_g):
            print('w', end=' ')
            for _ in range(repeat):
                print('r', end=' ')
                searcher = search_mtd(*plan.values())
                if i == 0:
                    with pos:
                        for _ in searcher:
                            count += 1
                with pos:
                    for _ in searcher:
                        pass
        results.append((count / region_num,
                        pos.t / (repeat * region_num)))
        print(results)
        return results


def run_all_exp(mtds_str, ds_path, ds_full_name, dsname, exps, ori_plan,
                q_meta, region_num, repeat):
    for framework, group in groupby(mtds_str, key=lambda x: x.split('_')[0]):
        match framework:
            case 'index':
                data_pack = build_index(ds_path, ds_full_name, cfg)
            case 'sliding':
                data_pack, _, _ = load_yolo_for(ds_path)
            case _:
                raise ValueError(f'impossible framework: {framework}!!!')
        for mtd_str in group:
            search_mtd = query_init(mtd_str, data_pack)
            for exp in exps:
                plans = func_m[exp](ori_plan, q_meta[exp], dsname)
                print(mtd_str, exp)
                run_one_exp(search_mtd, plans, region_num, repeat)


def run_grid_exp(mtds_str, refined_plan, grids, region_num, repeat):
    data_pack = build_index(ds_pth, ds_full, cfg)
    bor_m = load_ext_index_meta(os.path.join(cfg.INDEX.META_PATH,
                                             f'{ds_full}.json'))
    for m_s in mtds_str:
        search_mtd = query_init(m_s, data_pack)
        # run the (1, 1) one
        run_one_exp(search_mtd, [refined_plan], region_num, repeat)
        # run the rest
        for grid in mod_grid(grids):
            # rebuild index
            segs = load_seg(data_pack[1], ds_full, bor_m,
                            cfg.INDEX.META_PATH, grid)
            data_pack[0] = build_rest(data_pack[1], segs, bor_m)

            run_one_exp(search_mtd, [refined_plan], region_num, repeat)


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='all',
                        help="use the first letter to name a dateset, or 'all' for all dateset (see dataset.json)")
    exp_g = parser.add_mutually_exclusive_group()
    exp_g.add_argument('-e', '--experiment', type=str, default='all',
                        help="the first letter to name a type of experiment,"
                             "or 'all' for all for all experiments.")
    exp_g.add_argument('-g', '--grid', action='store_true', default=False,
                       help="run grid search experiment")
    parser.add_argument('-m', '--search-method', type=str, default='b',
                        help="use the first letter to name a type of framework or 's' for sliding, 'i' for index,"
                             "or 'b' for both framework")
    parser.add_argument('-o', '--output', type=str, default='stdout',
                        help="output file name, default is stdout")
    parser.add_argument('-c', '--config', type=str, default='',
                        help="costumed config file, default is 'config.json'")
    parser.add_argument('-n', '--region-number', type=int, choices=range(1, 4), default=3,
                        help="how many regions pre config")
    parser.add_argument('-r', '--repeat', type=int, choices=range(1, 11), default=1,
                        help="how many runs pre plan")
    return parser.parse_args()


if __name__ == '__main__':
    import os
    import json

    args = parse_arg()
    # dataset paths
    dataset_pth = os.path.join(os.path.dirname(__file__), 'query_configs')
    dataset = json.load(open(os.path.join(dataset_pth, 'dataset.json')))
    desired_ds = gather_all(args.dataset, abbr_map(dataset))
    ds_pths = [os.path.join(cfg.DATA.PATH, ds) for ds in full_name(dataset, desired_ds)]
    # search method
    mtds = which_mth(args.search_method)
    # experiment
    reg_num = args.region_number
    if args.config:
        query_meta = json.load(open(args.config))
    else:
        query_meta = json.load(open(os.path.join(dataset_pth, 'query_plans.json')))
    dft_plan = default_plan(query_meta['default'])
    grid_scale = query_meta.pop('grid_scale')
    if args.grid:
        cfg.BOX_NP = False
        cfg.INDEX.REGION.GRID.SPACE = (1, 1)
        # FIXME: query region should be small
        run_grid = True
    else:
        run_grid = False
        desired_exp = gather_all(args.experiment, abbr_map(query_meta))
        try:  # grid experiment are more complicated, treats it differently
            desired_exp.remove('grid_scale')
        except ValueError:
            pass
    for ds, ds_pth in zip(desired_ds, ds_pths):
        re_plan = copy.deepcopy(dft_plan)
        regions = query_meta['region']['bboxes'][ds]
        refine_region(re_plan, regions)
        refine_interval(re_plan, query_meta)
        ds_full = full_name(dataset, [ds])[0]
        if run_grid:
            for m in mtds:
                if m.startswith('sliding'):
                    raise ValueError('grid search do not support sliding based')
            run_grid_exp(mtds, re_plan, grid_scale, reg_num, args.repeat)
        else:
            run_all_exp(mtds, ds_pth, ds_full, ds, desired_exp, re_plan,
                        query_meta, reg_num, args.repeat)
