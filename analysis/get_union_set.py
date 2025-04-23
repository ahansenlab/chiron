import pandas as pd
import os
import math
import numpy as np

def mindist(t, U):
    ds = [math.dist(t, UU) for UU in U]
    min_val = np.min(ds)
    min_U = U[np.nonzero(ds == min_val)[0][0]]

    out = [t, min_U, min_val]

    return out

def get_distances(called_df, truth_df):
    called_coords = list(zip(called_df['start1'], called_df['end1']))
    truth_coords = list(zip(truth_df['start1'], truth_df['end1']))

    dist_tbl = pd.DataFrame(columns=['c1', 'c2', 'distance'])
    for i, t in enumerate(truth_coords):
        # find the called loop with minimum distance from the true loop
        dist_tbl.loc[i] = mindist(t, called_coords)

    return dist_tbl

def get_distances_chr(df1, df2):
    dist_tbl = pd.DataFrame(columns=['c1', 'c2', 'distance'])
    df1['mindist'] = np.nan
    df2['mindist'] = np.nan
    for chrom in df1['chr1'].unique():
        print(chrom)
        df1_chr = df1[df1['chr1']==chrom]
        df2_chr = df2[df2['chr1'] == chrom]
        dist_df12_chr = get_distances(df2_chr, df1_chr)  # c1 is df1 here; dist frm each df1
        dist_df21_chr = get_distances(df1_chr, df2_chr) # c1 is df2 here; dist frm each df2
        # match both dfs based on having coordinates in the union

        df1.loc[df1[df1['chr1']==chrom].index,'mindist'] = dist_df12_chr['distance'].values
        df2.loc[df2[df2['chr1'] == chrom].index,'mindist'] = dist_df21_chr['distance'].values

    return df1, df2

# read both sets of loopcalls
mus_dir = "/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/mustache"
chiron_dir = '/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/fracshift_annotations/'

mustache_df = pd.read_csv(os.path.join(mus_dir, 'GM12878_mustache.bedpe'), sep='\t',
                          names=['BIN1_CHR', 'BIN1_START', 'BIN1_END', 'BIN2_CHR', 'BIN2_START', 'BIN2_END', 'FDR'])
binsize = mustache_df['BIN1_END'] - mustache_df['BIN1_START']
binsize = binsize.iloc[0]
mustache_df.drop(['BIN1_END', 'BIN2_END', 'BIN2_CHR'], axis=1, inplace=True)
mustache_df.rename({'BIN1_CHR': 'chr1', 'BIN1_START': 'start1', 'BIN2_START': 'end1', 'FDR': 'prob'},
                   axis=1,
                   inplace=True)
mustache_df['start1'] = mustache_df['start1'] + binsize
mustache_df['end1'] = mustache_df['end1'] + binsize

chiron_df = pd.read_csv(os.path.join(chiron_dir, 'GM12878_loops_fracshift_200_mask3_all.txt'), sep='\t')
print(chiron_df)
# compute minimum distance between them and sort for distance less than 3kb in Euclidean distance
mus_dist, chiron_dist = get_distances_chr(mustache_df, chiron_df)
mus_valid = mus_dist[mus_dist['mindist']<=3000].drop('mindist', axis=1)
chiron_valid = chiron_dist[chiron_dist['mindist']<=3000].drop('mindist', axis=1)

print(f"Number of CHIRON loops: {len(chiron_valid)}")
print(f"Number of mustache loops: {len(mus_valid)}")

# get the df of tuple coordinates to match the originals
mus_valid.to_csv('/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/mus_within_chiron.bedpe', sep='\t', header=None, index=None)
chiron_valid.to_csv('/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/chiron_within_mus.bedpe', sep='\t', header=None, index=None)