'''
Written by Fan Feng
Edited/point of contact: Varshini Ramanathan (varsh@mit.edu)
Purpose: generate training data for CHIRON
'''

import os
import numpy as np
import cooler
from cooltools.lib import numutils
from scipy import sparse, linalg
import seaborn as sns
import matplotlib.pyplot as plt

def get_matr(clr, capture_string):
    clr_mat = clr.matrix(balance=True).fetch(capture_string)
    clr_mat[np.isnan(clr_mat)] = 0
    clr_mat_balanced = numutils.observed_over_expected(clr_mat)[0]
    return clr_mat_balanced

def load_all_loops(files, resolution=1000):
    loops = {}
    for file in files:
        for line in open(file):
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

def load_all_loops_bedform(files, resolution=1000, pad = 1000):
    loops = {}

    ct = 0
    for line in open(files):
        if ct == 0:
            ct +=1
            continue
        else:
            lst = line.strip().split()[:3]
            if lst == []:
                continue # empty lines in text file
            else:

                ch = lst[0]
                if (lst[1] == 'anchor1') | (lst[1] == 'start1'):  # hacky way of bypassing first line
                    continue

                if not ch.startswith('chr'):
                    ch = 'chr' + ch
                if ch not in loops:
                    loops[ch] = []

                p1 = int(float(lst[1]))
                p2 = int(float(lst[2]))

                # p11, p12, p21, p22 = p1-pad, p1+pad, p2-pad, p2+pad
                # c1, c2 = (p11 + p12) // 2 // resolution, (p21 + p22) // 2 // resolution

                c1, c2 = p1 // resolution, p2 //resolution
                if c1 > c2:
                    c1, c2 = c2, c1
                loops[ch].append((c1, c2))

    for ch in loops:
        print(ch, len(loops[ch]))
        loops[ch] = sorted(loops[ch])
    return loops

def load_all_regions(output_path, clr_path, ch_coord, resolution=1000):
    n_distance = 2000000 // resolution
    distance_sum = np.zeros((n_distance,))
    length = 0

    matrices = {}
    clr = cooler.Cooler(f'{clr_path}::resolutions/{str(resolution)}')


    for ch in ch_coord:
        if not is_nested_tuple(ch_coord[ch]):
            st, ed = ch_coord[ch]
            length += (ed - st) // resolution
            size = (ed - st) // resolution
            print(st, ed)
            capture_string = ch + ":" + str(st) + "-" + str(ed)
            mat = get_matr(clr, capture_string)
            matrices[ch] = mat
        else:
            print(f"multiple ROIs for {cell_type} at {ch}")
            for i, coords in enumerate(ch_coord[ch]):
                st, ed = coords
                print(st, ed)
                length += (ed - st) // resolution
                size = (ed - st) // resolution

                capture_string = ch + ":" + str(st) + "-" + str(ed)
                print(capture_string)
                mat = get_matr(clr, capture_string)
                matrices[f"{ch}_{i}"] = mat

    lengths = np.array([length - i * len(ch_coord) for i in range(n_distance)])
    distance_decay = distance_sum / lengths
    np.savetxt(output_path + 'RCMC_distance_decay.txt', distance_decay)
    return matrices

def is_nested_tuple(item):
     return any(isinstance(c, tuple) for c in item)


def region_generator(clr_path, cell_type, ch_coord, loop_files, output_path, window_size=15, resolution=1000):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + '/pos/'):
        os.mkdir(output_path + '/pos/')

    # Load Micro-C strata
    matrices = load_all_regions(output_path, clr_path, ch_coord)

    # Convert to output contact maps
    loops = load_all_loops_bedform(loop_files)

    # Convert to output contact maps
    cnt = 0

    for ch in loops:
        if ch in ch_coord.keys():
            if not is_nested_tuple(ch_coord[ch]):
                st, ed = ch_coord[ch]
                print(st, ed)
                new_st, new_ed = st // resolution, ed // resolution
                for (c1, c2) in loops[ch]:
                    if cnt % 100 == 0:
                        print(f' Loop: {cnt}')
                    if c1 - window_size >= new_st and c2 - window_size >= new_st and c1 + window_size + 1 < new_ed and c2 + window_size + 1 < new_ed:
                        cnt += 1
                        new_c1, new_c2 = c1 - new_st, c2 - new_st
                        mat = matrices[ch][new_c1 - window_size: new_c1 + window_size + 1,
                              new_c2 - window_size: new_c2 + window_size + 1]
                        ky = f'{cell_type}_{ch}_{c1}_{c2}'
                        np.save(f'{output_path}/pos/{ky}.npy', mat)
                        print(ky)

            else:
                print(f"multiple ROIs for {cell_type} at {ch}")
                for i, coords in enumerate(ch_coord[ch]):
                    st, ed = coords
                    print(st,ed)
                    new_st, new_ed = st // resolution, ed // resolution
                    for (c1, c2) in loops[ch]:
                        if cnt % 100 == 0:
                            print(f' Loop: {cnt}')

                        if c1 - window_size >= new_st and c2 - window_size >= new_st and c1 + window_size + 1 < new_ed and c2 + window_size + 1 < new_ed:
                            cnt += 1
                            new_c1, new_c2 = c1 - new_st, c2 - new_st
                            mat = matrices[f"{ch}_{i}"][new_c1 - window_size: new_c1 + window_size + 1, new_c2 - window_size: new_c2 + window_size + 1]
                            ky = f'{cell_type}_{ch}_{c1}_{c2}'
                            np.save(f'{output_path}/pos/{ky}.npy', mat)
                            print(ky)

    print(f"{cnt} total loops for {cell_type}")


if __name__ == '__main__':
    # viraat's original
    # viraat's mitosis
    # deep gm
    # fixed annotations for other CTs?
    # blood

    ch_coords = {'GM12878_og':{
        'chr6': (25110000, 28630000),
        'chr1': (207620000, 210340000),
        'chr19': (36020000, 39760000)},

        'GM12878': {
            'chr6': (25110000, 28630000),
            'chr1': (207620000, 210340000),
            'chr19': (36020000, 39760000)},

        'HCT116': {
            'chr6': (25110000, 28630000),
            'chr1': (207620000, 210340000),
            'chr19': (36020000, 39760000)},

        'K562': {
            'chr6': (25110000, 28630000),
            'chr1': (207620000, 210340000),
            'chr19': (36020000, 39760000)},

        'H1': {
            'chr6': (25110000, 28630000),
            'chr1': (207620000, 210340000),
            'chr19': (36020000, 39760000)},

        # Ppm1g: chr5:31257344 - 32382344
        # Klf1: chr8:84846629 - 85856629
        # Fbn2: chr18:58032072 - 59034072
        # Sox2: chr3:33804149 - 35704149
        # Nanog: chr6:122451959 - 122876959

        'mm39og': {
            'chr8': (84120000, 85820000),
            'chr3': (33750000, 35650000),
            'chr5': (31100000, 32375000),
            'chr18': (57899000, 58900000),
            'chr6': (122450000, 122880000)
            },

        'mm39mitosis': {
            'chr2': (151920000, 153000000),
            'chr8': ((84840000, 85860000),(123000000, 124101000)),
            'chr9': (106675000, 108600000),
            'chr15': (61850000, 63684000)
            },

        'Eryth': {
            'chr2': (59960000, 60950000),
            'chr6': ((89230000, 90100000), (134130000, 135960000)),
            'chr11': (4460000, 6210000),
            'chr19': ((3730000, 4390000), (12650000, 13120000))}
    }
    #'GM12878_og': 'GM12878_merged_011424.50.mcool',
    clr_names = {'GM12878': 'GM12878_merged_realigned.50.mcool',
                 'HCT116': 'HCT116_merged_realigned.50.mcool',
                 'K562': 'K562_merged_realigned.50.mcool',
                 'H1': 'H1_merged_realigned.50.mcool',
                 'mm39og': 'RCMC_rev_allCap_DMSO_mm39.merged.50.mcool',
                 'mm39mitosis': 'RCMC_Blobel_AnaTelo_allCap_mm39.merged.50.mcool',
                 'Eryth': 'Blood_P123_merged.50.mcool'}

    cell_types = ['GM12878', 'HCT116', 'K562', 'H1', 'mm39og', 'mm39mitosis', 'Eryth']
    call_date = '092324'

    base_dir = '/home/varsh/LoopCaller/'
    clr_dir = '/pool001/varsh/rcmc/data/'
    loop_dir = f'rcmc_annotations/calls_{call_date}/'

    # have to be point calls
    loop_files = {
        'GM12878':'GM12878_092324.txt',
        'HCT116':'HCT116_092324.txt',
        'K562':'K562_092324.txt',
        'H1':'H1_092324.txt',
        'mm39og':'RCMC_2023_Loops_All.bed',
        'mm39mitosis':'Mitosis_LoopCalls_mm39.bed',
        'Eryth':'Blood_Calls_All.bed'
    }

    # assumes training same loci for all cell types. need to modify if not the case **
    for cell_type, clr in clr_names.items():
        ch_coord = ch_coords[cell_type]
        output_path = os.path.join(base_dir, 'loop_training_v2/')
        print(f"Beginning {cell_type}")
        loop_path = os.path.join(base_dir, loop_dir, loop_files[cell_type])  # merged loop file for given cell type
        clr_path = clr_dir + clr  # location of merged cooler for given cell type
        print(clr_path, flush=True)
        print(loop_path, flush=True)
        region_generator(clr_path, cell_type, ch_coord, loop_path, output_path, window_size=15)  # radius of the kernel

