import pandas as pd
import numpy as np
from cooltools.lib import numutils
import re


# no matter what the input format, generate a bed-style output: chr, st, ed, all other columns
def read_loopfile_to_bed(loopfile, other_names=None, skip_col=()):
    loop_df = pd.read_csv(loopfile, sep='\s+', header=None)
    #print(len(loop_df.columns))
    #print(loop_df)

    if other_names is not None:
        col_names = ['chr', 'start', 'end'].append(other_names)
    else:
        col_names = ['chr', 'start', 'end']

    if loopfile.endswith('bed'):
        loop_df.columns = col_names
        loop_df.drop(loop_df.columns[skip_col], axis=1, inplace=True)

        return loop_df

    elif loopfile.endswith('bedpe'):
        out_df = pd.DataFrame(columns=col_names)

        out_df['chr'] = loop_df.iloc[:,0]
        out_df['start'] = [int(x) for x in np.mean([np.array(loop_df.iloc[:,1]), np.array(loop_df.iloc[:,2])], axis=0)]
        out_df['end'] = [int(x) for x in np.mean([np.array(loop_df.iloc[:,4]), np.array(loop_df.iloc[:,5])], axis=0)]

        if len(skip_col)>0:
            assert np.min(skip_col)>=5, "Cannot skip any of the chr/st/ed columns"
            loop_df.drop(loop_df.columns[skip_col], axis=1, inplace=True)
        if other_names is not None:
            assert len(loop_df.columns) == 6+len(other_names), "Kept and skipped columns must add up to total columns"

            for i, ocol in enumerate(other_names):
                    out_df[ocol] = loop_df.iloc[:, 6+i]
        return out_df

    else:
        print("File format must be .bed or .bedpe")
        # ex. .txt file
        # flexible handling (not implemented yet)
        return []


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

def loops_to_matrix_bedpe(df, st, ed, binsize=1000):
    loop_coords = ((df[['start1', 'start2']].to_numpy()) + (binsize//2)- st) // binsize

    # initialize matrix based on the binsize and chrom coordinates
    matr_size = (ed - st) // binsize
    truth_matr = np.zeros([matr_size, matr_size])

    # fill the matrix with the loop coordinates
    # pad is in pixel coordinates for now (should prob translate to genomic BP) **
    for i in range(len(loop_coords)):
        truth_matr[int(loop_coords[i, 0]), int(loop_coords[i, 1])] = 1
    return truth_matr

def loops_to_matrix_bed(df, st, ed, binsize=1000):
    loop_coords = ((df[['anchor1', 'anchor2']].to_numpy()) - st) // binsize

    # initialize matrix based on the binsize and chrom coordinates
    matr_size = (ed - st) // binsize
    truth_matr = np.zeros([matr_size, matr_size])

    # fill the matrix with the loop coordinates
    # pad is in pixel coordinates for now (should prob translate to genomic BP) **
    for i in range(len(loop_coords)):
        truth_matr[int(loop_coords[i, 0]), int(loop_coords[i, 1])] = 1

    return truth_matr

def get_matr(clr, capture_string, oe=True):
    clr_mat = clr.matrix(balance=True).fetch(capture_string)
    clr_mat[np.isnan(clr_mat)] = 0
    if oe:
        clr_mat_balanced = numutils.observed_over_expected(clr_mat)[0]
        return clr_mat_balanced
    else:
        return clr_mat

def make_df_from_loops(loops, res, window=1):
    df = pd.DataFrame(columns=['# chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'prob'])
    w = window * res

    with open(loops, 'r') as f:
        first_line = f.readline()
        num_col = len(first_line.split())
        print(num_col)

        for line in f:
            lst = line.strip().split()[:num_col]

            if lst == []:
                continue  # empty lines in text file
            ch = lst[0]

            if (lst[1] == 'anchor1') | (lst[1] == 'start1'):  # hacky way of bypassing first line
                continue

            p1, p2 = int(float(lst[1])), int(float(lst[2]))
            if num_col > 3:
                pr = float(lst[3])
            else:
                pr = 1

            if p1 > p2:
                print(p1, p2)
                p1, p2 = p2, p1

            new_row = pd.DataFrame.from_dict(
                data={"# chr1": [ch], "start1": [p1 - w], "end1": [p1 + w], "chr2": [ch], "start2": [p2 - w],
                      "end2": [p2 + w], "prob": [pr]})
            df = pd.concat([df, new_row])

    #df_bed = BedTool.from_dataframe(df)

    return df
    # return df, df_bed

def make_df_from_loops_bedpe(loops):
    df = pd.DataFrame(columns=['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2'])

    with open(loops, 'r') as f:
        first_line = f.readline()
        num_col = len(first_line.split())
        for line in f:
            lst = line.strip().split()[:num_col]

            if lst == []:
                continue  # empty lines in text file
            ch = lst[0]

            if (lst[1] == 'anchor1') | (lst[1] == 'start1'):  # hacky way of bypassing first line
                continue

            p1, p2, p3, p4 = int(float(lst[1])), int(float(lst[2])), int(float(lst[4])), int(float(lst[5]))

            if p1 > p2:
                print(p1, p2)
                p1, p2 = p2, p1

            new_row = pd.DataFrame.from_dict(
                data={"chr1": [ch], "start1": [p1], "end1": [p2], "chr2": [ch], "start2": [p3],
                      "end2": [p4]})
            df = pd.concat([df, new_row])

    return df

def make_anchor_df(loops):
    df = pd.DataFrame(columns=['chr', 'anchor1', 'anchor2', 'prob'])

    with open(loops, 'r') as f:
        first_line = f.readline()
        num_col = len(first_line.split())
        print(num_col)

        for line in f:
            lst = line.strip().split()[:num_col]

            if lst == []:
                continue  # empty lines in text file
            ch = lst[0]

            p1, p2 = int(float(lst[1])), int(float(lst[2]))
            if num_col > 3:
                pr = float(lst[3])
            else:
                pr = 1

            if p1 > p2:
                print(p1, p2)
                p1, p2 = p2, p1

            new_row = pd.DataFrame.from_dict(
                data={"chr": [ch], "anchor1": [p1], "anchor2": [p2], "prob": [pr]})
            df = pd.concat([df, new_row])

    #df_bed = BedTool.from_dataframe(df)
    return df
    # return df, df_bed