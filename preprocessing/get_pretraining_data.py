'''
Written by Fan Feng
Edited/point of contact: Varshini Ramanathan (varsh@mit.edu)
Purpose: generate training data for CHIRON
'''

import os
import numpy as np
import cooler
from cooltools.lib import numutils
from cooltools.api import expected
from scipy import sparse, linalg
from multiprocessing import Pool
import pandas as pd

def get_matr(clr, capture_string=None):
    if capture_string is not None:
        clr_mat = clr.matrix(sparse=True, balance=True).fetch(capture_string)
    else:
        clr_mat = clr.matrix(sparse=True, balance=True)[:]

    clr_mat[np.isnan(clr_mat)] = 0
    clr_mat_balanced = numutils.observed_over_expected(clr_mat)[0]
    return clr_mat_balanced


def load_all_loops(files, resolution=1000):
    loops = {}

    for line in open(files):
        lst = line.strip().split()[:6]
        ch = lst[0]
        if not ch.startswith('chr'):
            ch = 'chr' + ch
        if ch not in loops:
            loops[ch] = []
        p11, p12, p21, p22 = int(lst[1]), int(lst[2]), int(lst[4]), int(lst[5])
        c1, c2 = (p11 + p12) // 2 // resolution, (p21 + p22) // 2 // resolution
        if c1 > c2:
            c1, c2 = c2, c1
        loops[ch].append((c1, c2))

    for ch in loops:
        print(ch, len(loops[ch]))
        loops[ch] = sorted(loops[ch])
    return loops

def get_matr_sparse(clr, cell_type, chr):
    print(f"Retrieving {chr}...")
    clr_mat_coo = clr.matrix(sparse=True, balance=True).fetch(chr)
    clr_mat_csr = clr_mat_coo.tocsr()

    return clr_mat_csr

# for whatever reason this hangs on the cluster
def load_all_regions_pool(clr, cell_type, chrs, resolution=1000):
    matrices = {}
    print("making pool..")
    pool = Pool()
    print("made pool...")
    args = [(clr, cell_type, ch) for ch in chrs]
    mat_list = []
    print("loading regions...")
    try:
        mat_list = pool.starmap(get_matr_sparse, args)
    finally:
        pool.close()
        pool.join()
    print("storing regions...")
    ct = 0
    if len(mat_list) > 0:
        for ch in chrs:
            matrices[ch] = mat_list[ct]
            ct += 1
    print("Done!")
    # for ch in chrs:
    #     print(f"Retrieving {ch}...")
    #     matrices[ch] = get_matr_sparse(clr, cell_type, ch)

    return matrices

def load_all_regions(clr, cell_type, chrs, resolution=1000):
    print("loading regions...")
    matrices = {}
    for ch in chrs:
        print(f"Retrieving {ch}...")
        matrices[ch] = get_matr_sparse(clr, cell_type, ch)

    print("Done!")
    return matrices

def get_expected(clr, cell_type):
    print(f"Computing expected values for {cell_type}...")
    expected_cols = expected.expected_cis(clr, ignore_diags=2)
    print(f"Generated expected values for {cell_type}!")
    return expected_cols

def region_generator(clr, output_path, cell_type, loop_files, expected_df, window_size=15, resolution=1000):
    # Load Micro-C strata
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + '/pos/'):
        os.mkdir(output_path + '/pos/')

    # Convert to output contact maps
    loops = load_all_loops(loop_files)  # struct by chromosome
    print("finished loading loops!")
    if cell_type == 'mESC':
        ch_all = ['chr' + str(i) for i in range(1, 20)]
    else:
        ch_all = ['chr' + str(i) for i in range(1, 23)]
    ch_all.append('chrX')

    matrices = load_all_regions(clr, cell_type, ch_all)

    # Convert to output contact maps
    cnt = 0
    for ch in ch_all:
        print(f"CHR{ch}")
        print(np.shape(matrices[ch]))

        expected_ch = expected_df[(expected_df['region1'] == ch) & (expected_df['region2'] == ch)]
        expected_vals = expected_ch['balanced.avg.smoothed.agg']

        st = 0
        ed = clr.chromsizes[ch] // resolution

        for (c1, c2) in loops[ch]:
            if c1 - window_size >= st and c2 - window_size >= st and c1 + window_size + 1 < ed and c2 + window_size + 1 < ed:
                mat_csr = matrices[ch][c1 - window_size: c1 + window_size + 1, c2 - window_size: c2 + window_size + 1]
                mat = mat_csr.toarray()
                mat[np.isnan(mat)] = 0
                # get cols rows which only correspond to c

                dnum = c2 - c1

                if dnum < (2*window_size + 1):
                    # create a mirrored toeplitz
                    diff = (2*window_size + 1) - dnum # how many diagonals to mirror
                    toep_col_valid = expected_vals.iloc[0:dnum][::-1]
                    rows_to_append = expected_vals.iloc[0:diff]
                    toep_col = pd.concat([toep_col_valid, rows_to_append], ignore_index = True)

                    toep_row = expected_vals.iloc[dnum:dnum + 2 * (window_size) + 1]
                    toep = linalg.toeplitz(toep_col, toep_row)


                else: # asymmetric toeplitz
                    # toep_col = expected_ch[expected_ch['dist'].between(dnum-2*(window_size), dnum)][::-1]
                    # toep_row = expected_ch[expected_ch['dist'].between(dnum, dnum+2*(window_size))]

                    toep_col = expected_vals.iloc[dnum-2*(window_size):dnum+1][::-1]
                    toep_row = expected_vals.iloc[dnum:dnum+2*(window_size)+1]

                    toep = linalg.toeplitz(toep_col, toep_row)

                    # indented this block because want to exclude close loops for now.
                    # so mat only saves if at least one window size away from diag
                    if np.shape(toep) != np.shape(mat):
                        print("BAD LOOP:")
                        print(dnum)
                        print(c1, c2)
                        print(np.shape(toep))
                        print("continuing...")
                    else:
                        mat_norm = np.divide(mat, toep)

                        if cnt % 100 == 0:
                            print(f' Loop: {cnt}')
                            print(np.shape(toep))
                            print(np.shape(mat))
                            print(f"dnum: {dnum}")
                            print(c1, c2)

                            # if cnt in [0, 1000, 5000]:
                            #     print(toep_col)
                            #     print(toep_row)
                            #     sns.heatmap(mat_norm, vmax=0.0001)
                            #     plt.show()
                            #     sns.heatmap(toep)
                            #     plt.show()
                        # make toeplitz from curr_cols dist from size c2-c1
                        # sum intensity and if below cutoff, don't save

                        ky = f'{cell_type}_{ch}_{c1}_{c2}'
                        np.save(f'{output_path}/pos/{ky}.npy', mat_norm)
                        #print(ky)
            cnt += 1


if __name__ == '__main__':
    ch_coord = None
    base_dir = '/home/varsh/LoopCaller/'
    loop_folder = base_dir + 'loop_pretraining_v2/pos'
    output_path = '/pool001/varsh/loop_pretraining_v2/'
    # output_path1 = base_dir + 'loop_pretraining/neg1'
    # output_path2 = base_dir + 'loop_pretraining/neg2'
    # output_path3 = base_dir + 'loop_pretraining/neg3'

    clr_names = {'mESC':'mESC_all_merged.mcool',
                 'HEK': 'HEK_UNT_COOLMERGED.250.mcool', 'H1': '4DNFI9GMP2J8.mcool'}
    loop_names = {'H1': 'H1_Mustache_01.txt', 'mESC': 'quantify_loops_1kb.tsv',
                  'HEK': 'HEK_UNT_COOLMERGED.proc.loops.2kb5kb10kb.tsv'}

    clr_dir = '/pool001/varsh/microc/'
    loop_dir = 'microc_annotations/'

    # assumes training same loci for all cell types. need to modify if not the case **
    for cell_type, clr_name in clr_names.items():
        curr_loop_name = loop_names[cell_type]
        loop_files = base_dir + loop_dir + curr_loop_name  # location of loop annotations for a given cell type
        clr_path = clr_dir + clr_name  # location of merged cooler for given cell type

        print(clr_path)
        print(loop_files)
        clr = cooler.Cooler(f'{clr_path}::resolutions/{str(1000)}')
        expfile = f"{base_dir}{cell_type}_expected.csv"

        if not os.path.isfile(expfile):
            expected_df = get_expected(clr, cell_type)
            expected_df.to_csv(expfile)
        else:
            expected_df = pd.read_csv(expfile)

        region_generator(clr, output_path, cell_type, loop_files, expected_df, window_size=15, resolution=1000)  # radius of the kernel

