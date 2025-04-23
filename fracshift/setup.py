# functions relating to setup and formatting
import utils
import numpy as np
import pandas as pd
import cooler
import matplotlib.pyplot as plt
from cooltools import numutils
# get k-by-k


# format loopcalls
def format_loopcalls(loops, capture_string, cutoffs, col_name='anno_res'):
    if col_name not in loops.columns:
        chrom, st, ed = utils.get_coords(capture_string)

        #distance of each loop from the diagonal, in kb
        dist_col = [((l['end1'])//1000)-((l['start1'])//1000) for r, l in loops.iterrows()]
        cutoff_df = pd.DataFrame.from_dict(cutoffs)
        cutoff_df.sort_values('cutoffs', inplace=True)

        res_col = [cutoff_df[cutoff_df['cutoffs'].ge(d) == True].iloc[0, 0] for d in dist_col]
        loops['distance'] = dist_col
        loops[col_name] = res_col
    #print(loops)
    loops['start1'] = loops['start1'].astype(int)
    loops['end1'] = loops['end1'].astype(int)
    return loops

def get_clr_mat(clr_path, capture_string, res):
    clr_path_res = f"{clr_path}::resolutions/{int(res)}"
    clr = cooler.Cooler(clr_path_res)
    clr_mat_balanced = get_matr(clr, capture_string)
    return clr_mat_balanced

def get_matr(clr, capture_string):
    clr_mat = clr.matrix(balance=True).fetch(capture_string)
    clr_mat[np.isnan(clr_mat)] = 0
    clr_mat_balanced = numutils.observed_over_expected(clr_mat)[0]
    return clr_mat_balanced

# def get_cooler_mats(clr_path, capture_string, res_list):
#     matr_all = {}
#     for res in res_list:
#         clr_path_curr = f"{clr_path}::resolutions/{int(res)}"
#         clr = cooler.Cooler(clr_path_curr)
#         matr_ub = clr.matrix(balance=True).fetch(capture_string)
#         matr_ub[np.isnan(matr_ub)] = 0
#         matr = numutils.observed_over_expected(matr_ub)[0]
#         #utils.plot_simple(matr)
#         matr_all[int(res)] = matr
#
#     return matr_all

def get_cooler_mats(clr_path, capture_string, res_list):
    matr_all = {}
    for res in res_list:
        matr_all[int(res)] = get_clr_mat(clr_path, capture_string, res)

    return matr_all

def get_loop_mats(cooler_mats, loops, capture_string, window=9, plotting=False):
    # subset loops at the resolution in anno_res at given window
    loop_mats = {}
    hw = (window) //2
    chrom, st, ed = utils.get_coords(capture_string)
    cc = 0
    for idx, loop in loops.iterrows():

        res = loop['anno_res']
        c1 = int(np.round((loop['start1'] - st) / res))
        c2 = int(np.round((loop['end1'] - st) / res))
        matr = cooler_mats[res]
        sub_matr = utils.get_sub_matr(matr, c1, c2, hw)
        #loop_mats[idx] = sub_matr
        loop_mats[cc] = sub_matr
        cc= cc+1
        #print(np.shape(sub_matr))
        if plotting:
            utils.plot(sub_matr, capture_string, res)

    return loop_mats

def make_res_list(cutoff_dict, target_list):
    c = cutoff_dict['res']
    res_list = list(set(c + target_list))
    return res_list
