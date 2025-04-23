'''
Written by Fan Feng
Edited/point of contact: Varshini Ramanathan (varsh@mit.edu)
Purpose: generate training data for CHIRON
'''

import os
import numpy as np
import cooler
from cooltools.lib import numutils
from multiprocessing import Pool
from cooltools.api import expected
from scipy import sparse, linalg
import pandas as pd

# def get_matr(clr, capture_string):
#     clr_mat = clr.matrix(balance=True).fetch(capture_string)
#     clr_mat[np.isnan(clr_mat)] = 0
#     clr_mat_balanced = numutils.observed_over_expected(clr_mat)[0]
#     return clr_mat_balanced
def load_all_loops(loop_folder, cell_type):
    distances = []
    loops = {}
    for _, __, files in os.walk(loop_folder):
        for file in files:
            name = file.split('.')[0]
            cname, ch, p1, p2 = name.split('_')
            if cname==cell_type:
                p1, p2 = int(p1), int(p2)
                if p1 > p2:
                    p1, p2 = p2, p1
                dis = p2 - p1
                distances.append(dis)

                if ch not in loops:
                    loops[ch] = []
                loops[ch].append((p1, p2))
        #print(loops.keys())
    return loops, distances, cname

def get_matr_sparse(clr, cell_type, chr):
    clr_mat_coo = clr.matrix(sparse=True, balance=True).fetch(chr)
    clr_mat_csr = clr_mat_coo.tocsr()

    return clr_mat_csr

def load_all_regions_legacy(clr, cell_type, chrs, resolution=1000):
    matrices = {}
    pool = Pool()
    args = [(clr, cell_type, ch) for ch in chrs]
    mat_list = []

    try:
        mat_list = pool.starmap(get_matr_sparse, args)
    finally:
        pool.close()
        pool.join()

    ct = 0
    if len(mat_list) > 0:
        for ch in chrs:
            matrices[ch] = mat_list[ct]
            ct += 1

    return matrices

def load_all_regions(clr, cell_type, chrs, resolution=1000):
    print("loading regions...")
    matrices = {}
    for ch in chrs:
        print(f"Retrieving {ch}...")
        matrices[ch] = get_matr_sparse(clr, cell_type, ch)

    print("Done!")
    return matrices

def get_expected(clr):
    expected_cols = expected.expected_cis(clr, ignore_diags=0)
    return expected_cols

def sample_loops1(clr, cell_type, expected_df, loop_folder, output_path, resolution=1000, window_size=15):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tol = 16
    # Two longest steps!!
    # get sparse matrices for every chromosome

    # get expected values for entire cooler
    print("loading loops for neg1...", flush=True)
    loops, distances, cname = load_all_loops(loop_folder, cell_type)
    print("finished loading loops!", flush=True)
    if cell_type == 'mESC':
        ch_all = ['chr' + str(i) for i in range(1, 20)]
    else:
        ch_all = ['chr' + str(i) for i in range(1, 23)]
    ch_all.append('chrX')

    matrices = load_all_regions(clr, cell_type, ch_all)

    np.random.seed(0)
    idx2coord = {}
    for ch in matrices:
        st = len(idx2coord)
        # each bin for each chromosome
        for i in range(clr.extent(ch)[1]-clr.extent(ch)[0]): # number of bins
            idx2coord[i + st] = (ch, i)
    cnt = 0
    choice_lst = np.arange(len(idx2coord))
    for dis in distances:
        found = False
        #print(dis)
        while not found:
            # randomly select from all bins
            idx = np.random.choice(choice_lst)
            ch, p1 = idx2coord[idx]

            expected_ch = expected_df[(expected_df['region1'] == ch) & (expected_df['region2'] == ch)]
            expected_vals = expected_ch['balanced.avg.smoothed.agg']

            # get a random point at dis
            p2 = dis + p1

            st = 0
            ed = clr.chromsizes[ch] // resolution

            actual_p1, actual_p2 = p1 + st, p2 + st
            overlap = False
            for (c1, c2) in loops[ch]:
                if abs(c1 - actual_p1) + abs(c2 - actual_p2) <= tol:
                    overlap = True
                    break

            if not overlap:
                if actual_p1 - window_size >= st and actual_p2 - window_size >= st and actual_p1 + window_size + 1 < ed and actual_p2 + window_size + 1 < ed:
                    mat_csr = matrices[ch][p1 - window_size: p1 + window_size + 1, p2 - window_size: p2 + window_size + 1]
                    mat = mat_csr.toarray()
                    mat[np.isnan(mat)] = 0

                    dnum = c2 - c1

                    if dnum < (2 * window_size + 1):
                        # create a mirrored toeplitz
                        diff = (2 * window_size + 1) - dnum  # how many diagonals to mirror
                        toep_col_valid = expected_vals.iloc[0:dnum][::-1]
                        rows_to_append = expected_vals.iloc[0:diff]
                        toep_col = pd.concat([toep_col_valid, rows_to_append], ignore_index=True)

                        toep_row = expected_vals.iloc[dnum:dnum + 2 * (window_size) + 1]
                        toep = linalg.toeplitz(toep_col, toep_row)

                    else:  # asymmetric toeplitz
                        toep_col = expected_vals.iloc[dnum - 2 * (window_size):dnum + 1][::-1]
                        toep_row = expected_vals.iloc[dnum:dnum + 2 * (window_size) + 1]
                        toep = linalg.toeplitz(toep_col, toep_row)

                        mat_norm = np.divide(mat, toep)

                        ky = f'{cell_type}_{ch}_{actual_p1}_{actual_p2}'
                        np.save(f'{output_path}/{ky}_{cnt}.npy', mat_norm)
                        cnt += 1

                    #print(ky)
                    found = True


    print(f"Total number of negative loops: {cnt}", flush=True)


def sample_loops2(clr, cell_type, expected_df, loop_folder, output_path, n_loops=500, resolution=1000, window_size=15, i=0):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tol = 16
    print("loading loops for neg2...")
    loops, distances, cname = load_all_loops(loop_folder, cell_type)
    print("finished loading loops!")
    if cell_type == 'mESC': # the mESC data doesn't have chr 20-22
        ch_all = ['chr' + str(i) for i in range(1, 20)]
    else:
        ch_all = ['chr' + str(i) for i in range(1, 23)]
    ch_all.append('chrX')

    matrices = load_all_regions(clr, cell_type, ch_all)
    np.random.seed(0)
    idx2coord = {}
    for ch in matrices:
        st = len(idx2coord)
        for i in range(clr.extent(ch)[1] - clr.extent(ch)[0]):  # number of bins
            idx2coord[i + st] = (ch, i)
    dis_lst = np.arange(2000)
    choice_lst = np.arange(len(idx2coord))


    for i in range(n_loops):
        found = False
        while not found:
            dis = np.random.choice(dis_lst)
            idx = np.random.choice(choice_lst)
            ch, p1 = idx2coord[idx]
            p2 = dis + p1

            expected_ch = expected_df[(expected_df['region1'] == ch) & (expected_df['region2'] == ch)]
            expected_vals = expected_ch['balanced.avg.smoothed']

            st = 0
            ed = clr.chromsizes[ch] // resolution

            actual_p1, actual_p2 = p1 + st, p2 + st
            overlap = False
            for (c1, c2) in loops[ch]:
                if abs(c1 - actual_p1) + abs(c2 - actual_p2) <= tol:
                    overlap = True
                    break

            if not overlap:
                if actual_p1 - window_size >= st and actual_p2 - window_size >= st and actual_p1 + window_size + 1 < ed and actual_p2 + window_size + 1 < ed:
                    mat_csr = matrices[ch][p1 - window_size: p1 + window_size + 1, p2 - window_size: p2 + window_size + 1]
                    mat = mat_csr.toarray()
                    mat[np.isnan(mat)] = 0

                    dnum = c2 - c1
                    if dnum < (2 * window_size + 1):
                        # create a mirrored toeplitz
                        diff = (2 * window_size + 1) - dnum  # how many diagonals to mirror
                        toep_col_valid = expected_vals.iloc[0:dnum][::-1]
                        rows_to_append = expected_vals.iloc[0:diff]
                        toep_col = pd.concat([toep_col_valid, rows_to_append], ignore_index=True)

                        toep_row = expected_vals.iloc[dnum:dnum + 2 * (window_size) + 1]
                        toep = linalg.toeplitz(toep_col, toep_row)

                    else:  # asymmetric toeplitz
                        toep_col = expected_vals.iloc[dnum - 2 * (window_size):dnum + 1][::-1]
                        toep_row = expected_vals.iloc[dnum:dnum + 2 * (window_size) + 1]
                        toep = linalg.toeplitz(toep_col, toep_row)
                        mat_norm = np.divide(mat, toep)

                        ky = f'{cname}_{ch}_{actual_p1}_{actual_p2}'
                        np.save(f'{output_path}/{ky}.npy', mat_norm)
                    #print(ky)
                    found = True


if __name__ == '__main__':
    ch_coord = None
    base_dir = '/home/varsh/LoopCaller/'
    data_dir = '/pool001/varsh'

    loop_folder = os.path.join(data_dir, 'loop_pretraining_v2/pos')
    output_path1 = os.path.join(data_dir, 'loop_pretraining_v2/neg1')
    output_path2 = os.path.join(data_dir, 'loop_pretraining_v2/neg2')

    clr_names = {'mESC': 'mESC_all_merged.mcool',
                 'HEK': 'HEK_UNT_COOLMERGED.250.mcool', 'H1': '4DNFI9GMP2J8.mcool'}

    clr_names = {'mESC': 'mESC_all_merged.mcool'}
    loop_names = {'H1': 'H1_Mustache_01.txt', 'mESC': 'quantify_loops_1kb.tsv',
                  'HEK': 'HEK_UNT_COOLMERGED.proc.loops.2kb5kb10kb.tsv'}

    clr_dir = '/pool001/varsh/microc/'
    loop_dir = 'microc_annotations/'

    ct = 0
    print("beginning...", flush=True)
    print(clr_dir)
    # assumes training same loci for all cell types. need to modify if not the case **
    for cell_type, clr_name in clr_names.items():
        print("loading cooler...", flush=True)
        clr_path = clr_dir + clr_name  # location of merged cooler for given cell type
        clr = cooler.Cooler(f'{clr_path}::resolutions/{str(1000)}')

        print(clr_path)

        expfile = f"{base_dir}{cell_type}_expected.csv"
        if not os.path.isfile(expfile):
            print("computing expected values...", flush=True)
            expected_df = get_expected(clr)
            expected_df.to_csv(expfile)
        else:
            expected_df = pd.read_csv(expfile)
        print(f"getting neg1 loops for {cell_type}...", flush=True)
        sample_loops1(clr, cell_type, expected_df, loop_folder, output_path1)
        #print(f"getting neg2 loops for {cell_type}...", flush=True)
        #sample_loops2(clr, cell_type, expected_df, loop_folder, output_path2, n_loops=5000, i = ct)
        ct += 1





