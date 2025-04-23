import re
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from cooltools.lib import plotting
from matplotlib import colors

def loops_to_matrix_bed(df, st, ed, binsize):
    loop_coords = ((df[['start1', 'end1']].to_numpy()) - st) // binsize

    # initialize matrix based on the binsize and chrom coordinates
    matr_size = (ed - st) // binsize
    truth_matr = np.zeros([matr_size, matr_size])

    # fill the matrix with the loop coordinates
    # pad is in pixel coordinates for now (should prob translate to genomic BP) **
    for i in range(len(loop_coords)):
        truth_matr[int(loop_coords[i, 0]), int(loop_coords[i, 1])] = 1

    return truth_matr

def bin_matr(mat, binsize):
    sz = int(np.shape(mat)[0] // binsize)
    blocks = mat.reshape((sz, int(binsize), sz, int(binsize)))
    mat_binned = np.sum(blocks, axis=(1, 3))
    return mat_binned

def get_sub_matr(matr, c1, c2, hw):
    sub_matr = matr[c1-hw:c1+hw+1,c2-hw:c2+hw+1]
    return sub_matr

def frac_round(pt, res_factor):
    c1 = np.round(pt[0] * res_factor) / res_factor
    c2 = np.round(pt[1] * res_factor) / res_factor
    return (c1, c2)
def plot(matr, capture_string, res):
    chrom, st, ed = get_coords(capture_string)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im2 = ax.matshow(matr[:], norm=colors.LogNorm(), cmap='fall')
    plt.colorbar(im2)

    labels = [(int(re.sub(u"\u2212", "-", item.get_text())) * res + st) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.show()
    pass

def plot_loops(matr, loops, capture_string, res, loops2=None):
    chrom, st, ed = get_coords(capture_string)
    loop_matr = loops_to_matrix_bed(loops, st, ed, res)

    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111)
    im2 = ax.matshow(matr[:], norm=colors.LogNorm(), cmap='fall')
    plt.colorbar(im2)

    predx, predy = np.nonzero(loop_matr)
    plt.scatter(predx, predy, s=8, edgecolors='b', facecolors='none', linewidth=1, alpha=1)

    if loops2 is not None:
        loop_matr2 = loops_to_matrix_bed(loops2, st, ed, res)
        predx2, predy2 = np.nonzero(loop_matr2)
        plt.scatter(predy2, predx2, s=8, edgecolors='k', facecolors='none', linewidth=1, alpha=1)



    labels = [(int(re.sub(u"\u2212", "-", item.get_text())) * res + st) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.show()
    pass

def plot_simple(matr, pt=None, ts= None, sd=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im2 = ax.matshow(matr[:], norm=colors.LogNorm(), cmap='fall')
    plt.colorbar(im2)
    if pt is not None:
        plt.scatter(pt[0], pt[1])
        # plt.scatter(0,0)
        # plt.scatter(1, 2)
    if ts is not None:
        plt.title(ts)

    if sd is not None:
        plt.savefig(sd)
    else:
        plt.show()
    pass

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


def make_default_cutoffs():
    # cutoffs[cutoffs] are in kb
    cutoffs = {'res': [1000, 2000, 5000], 'cutoffs': [500, 2000, np.inf]}
    return cutoffs
# distance computation function


