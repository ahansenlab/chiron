'''
Written by Fan Feng
Edited/point of contact: Varshini Ramanathan (varsh@mit.edu)
Purpose: generate training data for CHIRON
'''

import os
import numpy as np
import cooler
from cooltools.lib import numutils
import re
def get_matr(clr, capture_string):
    clr_mat = clr.matrix(balance=True).fetch(capture_string)
    clr_mat[np.isnan(clr_mat)] = 0
    clr_mat_balanced = numutils.observed_over_expected(clr_mat)[0]
    return clr_mat_balanced

def load_all_loops(loop_folder, cell_type):
    distances = []
    loops = {}
    # walk the folder which has all the positive data
    for _, __, files in os.walk(loop_folder):
        for file in files:
            name = file.split('.')[0]
            # according to the naming convention set in get_training_data_v2.py
            tags = re.split('_', name)
            cname = tags[0]
            ch = tags[1]
            p1 = tags[2]
            p2 = tags[3]
            # cname, ch, p1, p2 = name.split('_')

            ## TODO ##
            # edge case for the way i happened to name the mm39s - need to change for more flexible handling
            if not ch.startswith('chr'):
                print("pos files were not named properly. assuming format 'name1_name2_chr_p1_p2' ")
                cname1, cname2, ch, p1, p2 = name.split('_')
                cname = cname1+cname2
            else:
                cname1=cname

            # since the loop folder contains data from multiple coolers, sort for only those from the current
            if cname == cell_type:
                print(cname, ch, p1, p2)
                cname_real=cname
                p1, p2 = int(p1), int(p2)
                if p1 > p2:
                    p1, p2 = p2, p1
                dis = p2 - p1
                distances.append(dis)

                if ch not in loops:
                    loops[ch] = []
                loops[ch].append((p1, p2))
    #print(loops.keys())
    return loops, distances, cname_real


def load_all_regions(clr_path, ch_coord, resolution=1000):
    matrices = {}

    clr = cooler.Cooler(f'{clr_path}::resolutions/{str(resolution)}')

    for ch in ch_coord:
        st, ed = ch_coord[ch]
        size = (ed - st) // resolution

        capture_string = ch + ":" + str(st) + "-" + str(ed)

        #get the oe normalized matrix for the given region
        mat = get_matr(clr, capture_string)

        matrices[ch] = mat
    return matrices
    # for ch in ch_coord:
    #     st, ed = ch_coord[ch]
    #     size = (ed - st) // resolution
    #     mat = np.zeros((size, size))
    #     path = f'/nfs/turbo/umms-drjieliu/proj/4dn/data/RegionCaptureMicroC/mESC/processed_cooler_mm10/RCMC_mm10_{ch}.txt'
    #     for line in open(path):
    #         lst = line.strip().split()
    #         if len(lst) != 8:
    #             continue
    #         p1, p2, v = int(lst[1]), int(lst[4]), float(lst[7])
    #         if st <= p1 < ed and st <= p2 < ed:
    #             p1, p2 = (p1 - st) // resolution, (p2 - st) // resolution
    #             mat[p1, p2] += v
    #             if p1 != p2:
    #                 mat[p2, p1] += v
    #     matrices[ch] = mat
    # return matrices


def sample_loops1(clr_path, ch_coord, loop_folder, output_path, cell_type, resolution=1000, window_size=15):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    tol = 16

    matrices = load_all_regions(clr_path, ch_coord, resolution)
    # get all the loops from the file corresponding to that clr
    loops, distances, cname = load_all_loops(loop_folder, cell_type)
    print(f"{len(loops)} found")
    np.random.seed(0)
    idx2coord = {}

    # idx2coord creates a dictionary mapping unique indices to the indices of each chrom's matrix
    for ch in matrices:
        st = len(idx2coord)
        for i in range(len(matrices[ch])):
            idx2coord[i + st] = (ch, i)

    # every possible choice of loop
    choice_lst = np.arange(len(idx2coord))
    ct=0
    for dis in distances:
        found = False
        #print(dis)
        while not found:
            # randomly select a matrix index
            idx = np.random.choice(choice_lst)
            ch, p1 = idx2coord[idx]

            # get the coordinate at distance dis from that pixel
            p2 = dis + p1

            st, ed = ch_coord[ch]
            st, ed = st // resolution, ed // resolution

            actual_p1, actual_p2 = p1 + st, p2 + st
            overlap = False

            # if the random pixel is within tol distance of a loop, then terminate
            for (c1, c2) in loops[ch]:
                if abs(c1 - actual_p1) + abs(c2 - actual_p2) <= tol:
                    overlap = True
                    break

            # else its a genuine negative sample
            if not overlap:
                if actual_p1 - window_size >= st and actual_p2 - window_size >= st and actual_p1 + window_size + 1 < ed and actual_p2 + window_size + 1 < ed:
                    mat = matrices[ch][p1 - window_size: p1 + window_size + 1, p2 - window_size: p2 + window_size + 1]
                    ky = f'{cell_type}_{ch}_{actual_p1}_{actual_p2}'
                    np.save(f'{output_path}/{ky}.npy', mat)
                    #print(ky)
                    found = True
                    ct+=1

    print(f"{ct} distance-matched negative loops found for {cell_type}")


def sample_loops2(clr_path, ch_coord, loop_folder, output_path, cell_type, n_loops=500, resolution=1000, window_size=15, i=0, dis_range=1000):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    tol = 16

    matrices = load_all_regions(clr_path, ch_coord, resolution)
    loops, distances, cname = load_all_loops(loop_folder, cell_type)

    np.random.seed(i)
    idx2coord = {}
    for ch in matrices:
        st = len(idx2coord)
        for i in range(len(matrices[ch])):
            idx2coord[i + st] = (ch, i)
    choice_lst = np.arange(len(idx2coord))

    ## TODO ##
    # the distances were randomly chosen within 2000 pixels (2 Mb) - i changed because regions are not always this large
    # now it's a specifiable parameter that i set by default to 1 Mb
    dis_lst = np.arange(dis_range)

    ct=0
    for i in range(n_loops):
        found = False
        while not found:
            dis = np.random.choice(dis_lst)
            idx = np.random.choice(choice_lst)
            ch, p1 = idx2coord[idx]
            p2 = dis + p1

            st, ed = ch_coord[ch]
            st, ed = st // resolution, ed // resolution

            actual_p1, actual_p2 = p1 + st, p2 + st
            overlap = False
            for (c1, c2) in loops[ch]:
                if abs(c1 - actual_p1) + abs(c2 - actual_p2) <= tol:
                    overlap = True
                    break

            if not overlap:
                if actual_p1 - window_size >= st and actual_p2 - window_size >= st and actual_p1 + window_size + 1 < ed and actual_p2 + window_size + 1 < ed:
                    mat = matrices[ch][p1 - window_size: p1 + window_size + 1, p2 - window_size: p2 + window_size + 1]
                    ky = f'{cell_type}_{ch}_{actual_p1}_{actual_p2}'
                    np.save(f'{output_path}/{ky}.npy', mat)
                    #print(ky)
                    found = True
                    ct+=1

    print(f"{ct} random negative loops for {cell_type}")


if __name__ == '__main__':

    ch_coords_1 = {'GM12878_og': {
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

        'mm39og': {
            'chr8': (84120000, 85820000),
            'chr3': (33750000, 35650000),
            'chr5': (31100000, 32375000),
            'chr18': (57899000, 58900000),
            #'chr6': (122450000, 122880000)
            },

        'mm39mitosis': {
            'chr2': (151920000, 153000000),
            'chr8': (84840000, 85860000),
            'chr9': (106675000, 108600000),
            'chr15': (61850000, 63684000)
        },

        'Eryth': {
            'chr2': (59960000, 60950000),
            'chr6': (89230000, 90100000),
            'chr11': (4460000, 6210000),
            'chr19': (3730000, 4390000)}
    }

    ch_coords_2 = {
        'mm39mitosis': {
            'chr8': (123000000, 124101000)
        },

        'Eryth': {
            'chr6': (134130000, 135960000),
            'chr19': (12650000, 13120000)}
    }


    # 'GM12878_og': 'GM12878_merged_011424.50.mcool',
    clr_names_1 = {'GM12878': 'GM12878_merged_realigned.50.mcool',
                 'HCT116': 'HCT116_merged_realigned.50.mcool',
                 'K562': 'K562_merged_realigned.50.mcool',
                 'H1': 'H1_merged_realigned.50.mcool',
                 'mm39og': 'RCMC_rev_allCap_DMSO_mm39.merged.50.mcool',
                 'mm39mitosis': 'RCMC_Blobel_AnaTelo_allCap_mm39.merged.50.mcool',
                 'Eryth': 'Blood_P123_merged.50.mcool'}

    clr_names_2 = {'mm39mitosis': 'RCMC_Blobel_AnaTelo_allCap_mm39.merged.50.mcool',
                 'Eryth': 'Blood_P123_merged.50.mcool'}

    cell_types = ['GM12878', 'HCT116', 'K562', 'H1', 'mm39og', 'mm39mitosis', 'Eryth']
    call_date = '092324'

    base_dir = '/home/varsh/LoopCaller/'
    clr_dir = '/pool001/varsh/rcmc/data/'
    loop_dir = f'rcmc_annotations/calls_{call_date}/'

    # have to be point calls
    loop_files = {
        'GM12878': 'GM12878_092324.txt',
        'HCT116': 'HCT116_092324.txt',
        'K562': 'K562_092324.txt',
        'H1': 'H1_092324.txt',
        'mm39og': 'RCMC_2023_Loops_All.bed',
        'mm39mitosis': 'Mitosis_LoopCalls_mm39.bed',
        'Eryth': 'Blood_Calls_All.bed'
    }

    loop_folder = base_dir + 'loop_training_v2/pos'
    output_path1 = base_dir + 'loop_training_v2/neg1'
    output_path2 = base_dir + 'loop_training_v2/neg2'
    output_path3 = base_dir + 'loop_training_v2/neg3'

    # assumes training same loci for all cell types. need to modify if not the case **
    ct = 0
    for cell_type, clr in clr_names_1.items():
        ch_coord = ch_coords_1[cell_type]
        print(f"Beginning {cell_type}")
        loop_path = os.path.join(base_dir, loop_dir, loop_files[cell_type])  # merged loop file for given cell type

        clr_path = clr_dir + clr  # location of merged cooler for given cell type
        print(clr_path, flush=True)
        print(loop_path, flush=True)

        sample_loops1(clr_path, ch_coord, loop_folder, output_path1, cell_type)
        sample_loops2(clr_path, ch_coord, loop_folder, output_path2, cell_type, n_loops=500, i=ct)
        #sample_loops2(clr_path, ch_coord, loop_folder, output_path3, n_loops=500, i=ct)
        ct+= 1

    for cell_type, clr in clr_names_2.items():
        ch_coord = ch_coords_2[cell_type]
        print(f"Beginning {cell_type}")
        loop_path = os.path.join(base_dir, loop_dir, loop_files[cell_type])  # merged loop file for given cell type

        clr_path = clr_dir + clr  # location of merged cooler for given cell type
        print(clr_path, flush=True)
        print(loop_path, flush=True)

        sample_loops1(clr_path, ch_coord, loop_folder, output_path1, cell_type)
        ct+= 1

    print(f"Saved {ct} total negative loops")

