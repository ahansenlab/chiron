import os
import pandas as pd
import loop_utils_2 as loop_utils
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
def search_excl_base(loops, bases, pad=50, prob_cutoff=1):
    #print(loops)
    #print(bases)
    mask = pd.Series([True] * len(loops))

    # Vectorized comparison: iterate through df2 and mark df1 rows that fall in any df2 range
    for i in range(len(bases)):
        mask &= ~(((loops['start1'].between(bases.iloc[i]['start1']-pad, bases.iloc[i]['end1']+pad)) |
                  (loops['end1'].between(bases.iloc[i]['start1']-pad, bases.iloc[i]['end1']+pad))) &
                  (np.abs(bases.iloc[i]['end1'] - bases.iloc[i]['start1']) >= 1000))
    return mask

def clean_df(pred_df, bases_exclude, region_str):
    ch, st, ed = loop_utils.get_coords(region_str)

    pred_df = pred_df[pred_df['chr1'] == ch]
    pred_df = pred_df[pred_df['start1'] > st]
    pred_df = pred_df[pred_df['end1'] < ed].reset_index(drop=True)

    bases_exclude_chr = bases_exclude[bases_exclude['chr1'] == ch]
    ind_to_keep = search_excl_base(pred_df, bases_exclude_chr, pad=0)
    pred_filtered = pred_df[ind_to_keep].reset_index(drop=True)

    return pred_filtered

def get_distances(called_df, truth_df, tol=None):
    called_coords = list(zip(called_df['start1'], called_df['end1']))
    truth_coords = list(zip(truth_df['start1'], truth_df['end1']))

    dist_tbl = pd.DataFrame(columns=['c1', 'c2', 'distance'])
    for i, t in enumerate(truth_coords):
        # find the called loop with minimum distance from the true loop
        dist_tbl.loc[i] = mindist(t, called_coords)

    return dist_tbl


def mindist(t, U):
    ds = [math.dist(t, UU) for UU in U]
    min_val = np.min(ds)
    min_U = U[np.nonzero(ds == min_val)[0][0]]

    out = [t, min_U, min_val]

    return out
def get_threshold_classes_mus(pred_df, real, thresholds, dist=3000):
    # thr_cols = [f'thr_{np.round(x, 4)}' for x in thresholds]
    thr_dict = {}
    for t in thresholds:
        pred = pred_df[pred_df['prob']<t]

        if len(pred) > 0:
            if len(real) == 0:
                TP = 0
                FP = len(pred)
                FN = 0
            else:

                pred2real = get_distances(pred, real)
                # finds the real loop with mindist from the pred loop
                real2pred = get_distances(real, pred)

                # there is a real loop near the pred loop
                TP = len(real2pred[real2pred['distance'] <= dist]['c1'].apply(pd.Series))
                # there is no real loop near the pred loop
                FP = len(real2pred[real2pred['distance'] > dist]['c1'].apply(pd.Series))
                # there is no pred loop near the real loop
                FN = len(pred2real[pred2real['distance'] > dist]['c1'].apply(pd.Series))

                precision = TP / (TP+FP)
                recall = TP / (TP+FN)

                print(precision, recall)
        else:
            print("no predictions")
            precision = 0
            recall = 0

            TP = 0
            FP = np.nan
            FN = len(real)

        thr_dict[t] = [TP, FP, FN]
    out = pd.DataFrame.from_dict(thr_dict)
    out.index = ['TP', 'FP', 'FN']
    return out

def get_threshold_classes(pred_df, real, thresholds, dist=3000):
    # thr_cols = [f'thr_{np.round(x, 4)}' for x in thresholds]
    thr_dict = {}
    for t in thresholds:
        pred = pred_df[pred_df['prob']>t]

        if len(pred) > 0:
            if len(real) == 0:
                TP = 0
                FP = len(pred)
                FN = 0
            else:
                pred2real = get_distances(pred, real)
                # finds the real loop with mindist from the pred loop
                real2pred = get_distances(real, pred)

                # there is a real loop near the pred loop
                TP = len(real2pred[real2pred['distance'] <= dist]['c1'].apply(pd.Series))
                # there is no real loop near the pred loop
                FP = len(real2pred[real2pred['distance'] > dist]['c1'].apply(pd.Series))
                # there is no pred loop near the real loop
                FN = len(pred2real[pred2real['distance'] > dist]['c1'].apply(pd.Series))

                precision = TP / (TP+FP)
                recall = TP / (TP+FN)
        else:
            precision = 0
            recall = 0

            TP = 0
            FP = np.nan
            FN = len(real)

        thr_dict[t] = [TP, FP, FN]
    out = pd.DataFrame.from_dict(thr_dict)
    out.index = ['TP', 'FP', 'FN']
    return out


if __name__ == '__main__':
    base_dir = '/mnt/md0/varshini/RCMC_LoopCaller/'

    celltypes = ['HCT116', 'H1', 'K562', 'GM12878']
    pred_dir = base_dir + ('loopcalls/v2_testing/pre_hp_4_ftr_2024-11-13/merged-strict/')
    mus_dir = base_dir + ('loopcalls/mustache/')
    call_dir = base_dir + 'loopcalls/Annotations_092324/'

    bases_exclude = pd.read_csv(os.path.join(base_dir, 'final_panel_bases_not_covered.bed'), sep='\s+')
    bases_exclude.rename({'chr': 'chr1', 'base1': 'start1', 'base2': 'end1'}, axis=1, inplace=True)
    bases_exclude['prob'] = 1

    ch_coord = {
        'region3': 'chr5:157000000-160150000',
        'region5': 'chr4:61369000-64435000'
    }

    # the mustache predictions are 0 below fdr 10e-5 so i started there
    thresholds_mus = np.logspace(-5, 0, 50)

    # the chiron loop probs are much more representative at high probs i.e. the # of predictions changes a lot at the tail
    # so i used a reverse log spacing to define the thresholds. the opposite is true for mustache hence the log spacing
    thresholds = 1000 - np.geomspace(1, 1000, 50)
    thresholds = preprocessing.minmax_scale(thresholds, feature_range=(0.7, 1), axis=0, copy=True)

    cols_og = ['chr1', 'start1', 'end1', 'prob']
    thr_cols = np.round(thresholds, 4)
    cols_og.extend(thr_cols)

    pred_all = pd.DataFrame(columns=np.round(thresholds, 4), index=['TP', 'FP', 'FN'])
    pred_all_mus = pd.DataFrame(columns=np.round(thresholds_mus, 4), index=['TP', 'FP', 'FN'])

    for ct in celltypes:
        print(ct)
        if ct =='GM12878':
            real_df = pd.read_csv(os.path.join(call_dir, f'GM12878_heldout_added_calls.txt'), sep='\s+')
        else:
            real_df = pd.read_csv(os.path.join(call_dir, f'{ct}_092324.txt'), names=['chr1', 'start1', 'end1'], sep='\s+')

        real_df['prob'] = 1
        for reg in ch_coord:
            print(reg)
            ch, st, ed = loop_utils.get_coords(ch_coord[reg])
            print(ch)

            # load the predicted loop calls (idli)
            pred_df = pd.read_csv(os.path.join(pred_dir, f'{ct}_region3,region5_loops_merged_gaussian.txt'), sep='\s+')
            pred_df.rename({'#chr': 'chr1', 'anchor1': 'start1', 'anchor2': 'end1', 'loopLikelihood': 'prob'}, axis=1,
                           inplace=True)
            pred_df['chr1'] = ch
            pred_filtered = clean_df(pred_df, bases_exclude, ch_coord[reg])

            mustache_df = pd.read_csv(os.path.join(mus_dir, f'{ct}_1kb_{ch}_s01.6_st0.88.tsv'), sep='\t')
            binsize = mustache_df['BIN1_END'] - mustache_df['BIN1_START']
            binsize = binsize.iloc[0]
            print(binsize)
            mustache_df.drop(['BIN1_END', 'BIN2_END', 'BIN2_CHROMOSOME', 'DETECTION_SCALE'], axis=1, inplace=True)
            mustache_df.rename({'BIN1_CHR': 'chr1', 'BIN1_START': 'start1', 'BIN2_START': 'end1', 'FDR': 'prob'},
                               axis=1,
                               inplace=True)
            mustache_df['start1'] = mustache_df['start1'] + binsize
            mustache_df['end1'] = mustache_df['end1'] + binsize
            mustache_df['chr1'] = ch
            mus_filtered = clean_df(mustache_df, bases_exclude, ch_coord[reg])

            #real_filtered = real_df[real_df['chr1']==ch]
            real_filtered = clean_df(real_df, bases_exclude, ch_coord[reg])

            # assign classes at a range of thresholds
            thr_df = get_threshold_classes(pred_filtered, real_filtered, np.round(thresholds, 4))
            pred_all = thr_df.add(pred_all, fill_value=0)

            thr_df_mus = get_threshold_classes_mus(mus_filtered, real_filtered, np.round(thresholds_mus, 4))
            pred_all_mus = thr_df_mus.add(pred_all_mus, fill_value=0)

    pred_all_mus.to_csv('/mnt/md0/varshini/RCMC_LoopCaller/pred_thresholds_mus_repro.txt', sep='\t')
    pred_all.to_csv('/mnt/md0/varshini/RCMC_LoopCaller/pred_thresholds_idli_repro.txt', sep='\t')






