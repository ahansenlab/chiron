from pybedtools import BedTool
import pandas as pd
import numpy as np
import cooler
from cooltools.lib import numutils, plotting
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import morphology, generate_binary_structure, binary_fill_holes
import cv2
import re
import math
import os
def get_matr(clr, capture_string):
    clr_mat = clr.matrix(balance=True).fetch(capture_string)
    clr_mat[np.isnan(clr_mat)] = 0
    clr_mat_balanced = numutils.observed_over_expected(clr_mat)[0]
    return clr_mat_balanced


def bedpe_to_coords(bedpe_df):
    coord1 = np.mean(np.array(bedpe_df[['start1', 'start2']]), axis=1)
    coord2 = np.mean(np.array(bedpe_df[['end1', 'end2']]), axis=1)
    coords = list(zip(coord1, coord2))  # list of tuples
    return coords

def get_distances(called_df, truth_df, tol=None):
    called_coords = list(zip(called_df['start1'], called_df['end1']))
    truth_coords = list(zip(truth_df['start1'], truth_df['end1']))

    dist_tbl = pd.DataFrame(columns=['true_coord', 'called coord', 'distance'])
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

def get_coords(locus_str):
    m = re.search(r'(chr\w+):(\d+)-(\d+)', locus_str)
    chrom = m.group(1)
    start_coord = int(m.group(2))
    end_coord = int(m.group(3))
    try:
        chrom = int(chrom)
    except:
        return chrom, start_coord, end_coord
    return chrom, start_coord, end_coord

def plot_overlaps(celltype, im, ch, st, ed, real, pred, mask=None, res=1000):
    lw = 0.8
    # plot

    # finds the pred loop with mindist from the real loop
    pred2real = get_distances(pred, real)

    # finds the real loop with mindist from the pred loop
    real2pred = get_distances(real, pred)


    # print(pred2real)
    # there is a real loop near the pred loop
    TP = real2pred[real2pred['distance'] <= 5000]['true_coord'].apply(pd.Series)
    # there is no loop near the pred loop
    FP = real2pred[real2pred['distance'] > 5000]['true_coord'].apply(pd.Series)
    # there is no pred loop near the real loop
    FN = pred2real[pred2real['distance'] > 5000]['true_coord'].apply(pd.Series)

    # print(TP)
    print(f"TP: {len(TP)} "
          f"FP: {len(FP)} "
          f"FN: {len(FN)} ")

    print(f"Precision: {len(TP)/(len(TP)+len(FP))} "
          f"Recall: {len(TP)/(len(TP)+len(FN))} ")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im2 = ax.matshow(im[:], norm=colors.LogNorm(), cmap='fall')

    real_matr = loops_to_matrix_2(real, st, ed, res)
    pred_matr = loops_to_matrix_2(pred, st, ed, res)
    pred_matr[mask == 0] = 0

    realx, realy = np.nonzero(real_matr)
    plt.scatter(realy, realx, s=4, edgecolors='k', facecolors='none', linewidth=lw, alpha=1)

    predx, predy = np.nonzero(pred_matr)
    plt.scatter(predx, predy, s=4, edgecolors='#0077BB', facecolors='none', linewidth=lw, alpha=1)

    plt.title(celltype)

    plt.show()

    # fig = plt.figure(figsize=(20, 20))
    # ax = fig.add_subplot(111)
    # im2 = ax.matshow(im[:], norm=colors.LogNorm(), cmap='fall')
    #
    # plt.scatter((TP.iloc[:,0]-st)//res, (TP.iloc[:,1]-st)//res, s=4, edgecolors='k', facecolors='none', linewidth=lw, alpha=1)
    # plt.scatter((FP.iloc[:,0]-st)//res, (FP.iloc[:,1]-st)//res, s=4, edgecolors='#0077BB', facecolors='none', linewidth=lw, alpha=1)
    # plt.scatter((FN.iloc[:,0]-st)//res, (FN.iloc[:,1]-st)//res, s=4, edgecolors='g', facecolors='none', linewidth=lw, alpha=1)
    # plt.legend(['TP','FP','FN'])

    # plt.show()
    # print(pred['start1'])
    # print(TP.iloc[:,0])

    TP_loops1 = pred[pred['start1'].isin(TP.iloc[:,0]) & pred['end1'].isin(TP.iloc[:,1])].reset_index(drop=True)
    FP_loops1 = pred[pred['start1'].isin(FP.iloc[:,0]) & pred['end1'].isin(FP.iloc[:,1])].reset_index(drop=True)
    pp=np.percentile(pred['prob'], 50)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # plt.hist(TP_loops1['prob'], bins = 70, alpha = 0.8, density = True)
    # plt.hist(FP_loops1['prob'], bins=70, alpha = 0.8, density = True)
    # plt.plot(np.repeat(pp, 100), np.linspace(0, plt.gca().get_ylim()[1], 100) ,'b--', alpha=0.5)
    # plt.plot(np.repeat(pp, 100), np.linspace(0, plt.gca().get_ylim()[1], 100), 'r--', alpha=0.5)
    # plt.legend([f"{np.round(len(TP_loops1[TP_loops1['prob'] > pp])/len(TP_loops1),4)} Loops Conserved",
    #             f"{np.round(len(FP_loops1[FP_loops1['prob'] < pp])/len(FP_loops1),4)} Loops Filtered",'TP','FP'])
    # plt.title(celltype)
    # plt.xlabel('Prediction Confidence')
    # plt.ylabel('Probability Density (A.U.)')
    # plt.show()
    #
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.hist(TP_loops1['prob'], bins=70, alpha=0.6)
    plt.hist(FP_loops1['prob'], bins=70, alpha=0.6)
    plt.legend(['TP', 'FP'])
    plt.title(celltype)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Counts')
    plt.show()

    pass


def loops_to_matrix_2(df, st, ed, binsize=1000):
    loop_coords = ((df[['start1', 'end1']].to_numpy()) - st) // binsize

    # initialize matrix based on the binsize and chrom coordinates
    matr_size = (ed - st) // binsize
    truth_matr = np.zeros([matr_size, matr_size])

    # fill the matrix with the loop coordinates
    # pad is in pixel coordinates for now (should prob translate to genomic BP) **
    for i in range(len(loop_coords)):
        truth_matr[int(loop_coords[i, 0]), int(loop_coords[i, 1])] = 1
    return truth_matr

def plot_overlaps_no_gt(celltype, im, ch, st, ed, pred_in, res, mask=None):
    # unpack
    pp = 0.999
    #pred = pred_in[pred_in['prob']>pp]
    pred=pred_in

    lw = 0.8
    # plot
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    im2 = ax.matshow(im[:], norm=colors.LogNorm(vmin=0.01), cmap='fall')

    truth_matr = loops_to_matrix_2(pred_in, st, ed, res)
    if mask is not None:
        truth_matr[mask==0] = 0
    predx, predy = np.nonzero(truth_matr)

    plt.scatter(predx, predy, s=4, edgecolors='b', facecolors='none', linewidth=lw, alpha=1)

    plt.title(celltype)
    plt.colorbar(im2)

    labels = [(int(re.sub(u"\u2212", "-", item.get_text()))*1000+st) for item in ax.get_xticklabels()]
    #print(labels)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.show()
    pass

def edge_exclusion(im, pad=1):
    # find all the 0 pixels
    zero_mask = (im == 0).astype(np.uint8)

    # the hough line transform finds vertical lines first, so identify those
    vert_lines = cv2.HoughLinesP(zero_mask, rho=1, theta=1 * np.pi / 180, threshold=np.size(zero_mask, 1) - 10,
                                 minLineLength=np.size(zero_mask, 1) - 10, maxLineGap=0)
    vert_mask = np.zeros_like(zero_mask)
    if vert_lines is not None:
        for line in vert_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vert_mask, (x1, y1), (x2, y2), (255, 0, 0), 3)
        zero_mask[vert_mask == 1] = 0  # temp erase the vert lines so the hough transform finds horz lines

    horz_lines = cv2.HoughLinesP(zero_mask, rho=1, theta=1 * np.pi / 180, threshold=np.size(zero_mask, 1) - 100,
                                 minLineLength=np.size(zero_mask, 1) - 100, maxLineGap=100)
    zero_mask_filt = np.zeros_like(zero_mask)

    if horz_lines is not None:
        for line in horz_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(zero_mask_filt, (x1, y1), (x2, y2), (255, 0, 0), 3)
    if vert_lines is not None:
        for line in vert_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(zero_mask_filt, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # exclude diagonal
    sz = np.shape(im)[0]
    cv2.line(zero_mask_filt, (0, 0), (sz-1, sz-1), (255, 0, 0), 10)

    struct1 = generate_binary_structure(2, 2)
    dil_zeros = morphology.binary_dilation(zero_mask_filt, structure=struct1, iterations=pad)
    mask = ~dil_zeros

    plt.figure()
    plt.matshow(mask)
    plt.show()

    return mask

def search_excl_base(loops, bases, pad=50, prob_cutoff=0.999):
    #print(loops)
    #print(bases)
    mask = pd.Series([True] * len(loops))

    # Vectorized comparison: iterate through df2 and mark df1 rows that fall in any df2 range
    for i in range(len(bases)):
        mask &= ~(((loops['start1'].between(bases.iloc[i]['start1']-pad, bases.iloc[i]['end1']+pad)) |
                  (loops['end1'].between(bases.iloc[i]['start1']-pad, bases.iloc[i]['end1']+pad))) &
                  (loops['prob'] < prob_cutoff))
    return mask

if __name__ == '__main__':
    base_dir = '/mnt/md0/varshini/RCMC_LoopCaller/'

    clr_names = {'HCT116': 'HCT116_merged_realigned.50.mcool',
                 'K562':'K562_merged_realigned.50.mcool', 'H1':'H1_merged_realigned.50.mcool'}
    clr_names = {'GM12878': 'GM12878_merged_realigned.50.mcool'}
    pred_dir = base_dir + ('loopcalls/v2_testing/impute_ftr_10-26/merged-strict/')
    #pred_dir = base_dir + ('loopcalls/peakachu/merged-strict/')
    #pred_dir = base_dir + ('imputation_ft012/chr5/')
    call_dir = base_dir + 'loopcalls/Annotations_092324/'
    bases_exclude = pd.read_csv(os.path.join(base_dir, 'final_panel_bases_not_covered.bed'),  sep='\s+')
    bases_exclude.rename({'chr': 'chr1', 'base1': 'start1', 'base2': 'end1'}, axis=1, inplace=True)
    bases_exclude['prob'] = 1

    clr_dir = 'data/'
    RES = 2000
    gt_exists=True

    ch_coord = {
        'region3': 'chr5:157000000-160150000',
        #'region5': 'chr4:61369000-64435000'
        #'region1': 'chr6:25111000-28622000'
        # 'region6': 'chr8:124988000-129796000',
        # 'region7': 'chr6:29678000-32257000',
        # 'region8': 'chrX:47081000-49443000',
        # 'region9': 'chr1:237286814-240531042', # in training set
        # 'region10': 'chr7:10480000-13507000',
        # 'region11': 'chr8:62918000-66567000',
        # 'region12': 'chr4:181289000-184015000',
        # 'region13': 'chr3:118624000-121534000',
        # 'region14': 'chr9:106603000-109910000'
    }
    # ** TODO SORT REAL LOOPS BY CHROM THAT WERE IMPUTED **

    # loop through files
    for ct, clr_name in clr_names.items():
        print(ct)
        clr_path = base_dir + clr_dir + clr_name
        clr = cooler.Cooler(f'{clr_path}::resolutions/{str(RES)}')

        for reg in ch_coord:
            if ct=='GM12878':
                ct2='GM'
            elif ct=='HCT116':
                ct2='HCT116'
            else:
                ct2=ct

            capture_string = ch_coord[reg]
            ch, st, ed = get_coords(capture_string)

            pred_name = pred_dir + ct + f'_loops_merged_gaussian.txt'
            # pred_name = pred_dir + ct + f'_region9_loops_merged.txt'
            #pred_name = pred_dir + 'GM12878_loops_by_intensity.txt'
            loop_name = call_dir + ct + '_092324.txt'
            mustache_name = (f'/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/mustache/GM12878_2kb.tsv')

            mustache_df = pd.read_csv(mustache_name, sep='\t')
            binsize = mustache_df['BIN1_END'] - mustache_df['BIN1_END']
            binsize=binsize.iloc[0]
            #print(binsize)
            mustache_df.drop(['BIN1_END', 'BIN2_END', 'BIN2_CHROMOSOME',  'DETECTION_SCALE'], axis=1, inplace=True)
            mustache_df.rename({'BIN1_CHR':'chr1', 'BIN1_START':'start1', 'BIN2_START': 'end1', 'FDR':'prob'}, axis=1, inplace=True)
            mustache_df['start1'] = mustache_df['start1'] + binsize
            mustache_df['end1'] = mustache_df['end1'] + binsize
            mustache_df = mustache_df[mustache_df['prob'] < 0.05]
            mustache_df = mustache_df[mustache_df['start1'] > st]
            mustache_df = mustache_df[mustache_df['end1'] < ed]
            #print(mustache_df)

            pred_df = pd.read_csv(pred_name, sep='\s+')
            pred_df.rename({'#chr':'chr1', 'anchor1':'start1', 'anchor2':'end1', 'loopLikelihood':'prob'}, axis=1, inplace=True)
            pred_df['chr1'] = ch
            p_cutoff = np.percentile(pred_df['prob'], 0)
            p_cutoff=0.995
            pred_df = pred_df[pred_df['start1'] > st]
            pred_df = pred_df[pred_df['end1'] < ed]
            pred_df=pred_df[pred_df['prob']>p_cutoff].reset_index()

            # extract the exclude panel for the current chrom
            # it's already sorted, so just look for the insert place of every loop and see if it's in between either of its neighbors ranges
            # use bisect
            bases_exclude_chr = bases_exclude[bases_exclude['chr1']==ch]
            ind_to_keep = search_excl_base(pred_df, bases_exclude_chr, pad=0)
            pred_filtered = pred_df[ind_to_keep].reset_index(drop=True)
            ind_to_keep2 = search_excl_base(mustache_df.reset_index(), bases_exclude_chr, pad=0)
            mustache_df = mustache_df.reset_index()[ind_to_keep2].reset_index(drop=True)
            print(f"{len(pred_df) - len(pred_filtered)} rows removed")
            if gt_exists:
                real_df = pd.read_csv(loop_name, names=['chr1','start1','end1'], sep='\t')
                real_df['prob'] = 1
                real_df = real_df[real_df['chr1'] == ch]

            im = get_matr(clr, capture_string)
            #mask = edge_exclusion(im, 10)
            mat = get_matr(clr, capture_string)

            #get metrics

            #print(pred_filtered)
            #print(real_df)





            #plot_overlaps(ct, mat, ch, st, ed, real_df, mustache_df, res=RES, mask=None)
            #plot_overlaps(ct, mat, ch, st, ed, real_df, pred_filtered, res=RES, mask=None)
            #plot_overlaps_no_gt(ct, mat, ch, st, ed, pred_filtered, RES, mask=None)
            #plot_overlaps_no_gt(ct, mat, ch, st, ed, pred_df, RES, mask=None)
            # #plot_overlaps_no_gt(ct, mat, ch, st, ed, real_df, RES, None)
            plot_overlaps_no_gt(ct, mat, ch, st, ed, mustache_df, RES, None)


