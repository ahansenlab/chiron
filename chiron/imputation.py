"""
Author: Fan Feng
Edited/point of contact: Varshini Ramanathan (varsh@mit.edu)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import cooler
import argparse
from datetime import date
import re
from cooltools.lib import numutils
from scipy.ndimage import gaussian_filter

# helper to get an ICE-balnced, OE-normalized matrix from an input cooler and region
# option (default is off) to perform a Gaussian blur on the data
# only impute on smoothed data if you also trained on smoothed data
def get_matr(clr, capture_string, transform=None):
    clr_mat = clr.matrix(balance=True).fetch(capture_string)
    clr_mat[np.isnan(clr_mat)] = 0
    clr_mat_balanced = numutils.observed_over_expected(clr_mat)[0]

    if transform is not None:
        clr_mat_balanced = gaussian_filter(clr_mat_balanced, sigma=1)

    return clr_mat_balanced


def CNN_model(lr=0.001):
    """
    Build the 3-layer CNN model
    """
    model = models.Sequential()
    model.add(layers.Conv2D(8, (4, 4), activation='relu', input_shape=(31, 31, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(6, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(6, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')
    return model


def normalize_mat(mat, thr=0.98):
    """
    Normalize the contact matrix for each small region
    The values in the matrix will be normalized to [0, 1] by dividing np.quantile(mat, thr)
    The default thr (which is also the thr used during model training) is 0.98
    """
    max_val = np.quantile(mat, thr)
    mat = mat / (max_val + 1e-8)
    mat[mat > 1] = 1
    mat[mat < 0] = 0


    return mat

# helper to get chrom, start, and end from input ucsc string
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

def batch_load_test_mats_clr(ch_coord, max_distance, min_distance, clr_path, tns=None,
                             resolution=1000, interval=5000,
                             window_size=15, batch_size=10):
    """
    Load the contact matrices from .mcool file,
     and return the 31*31 regions in batches for the model to make predictions
    """
    clr = cooler.Cooler(f'{clr_path}::resolutions/{str(resolution)}')

    for ch in ch_coord:
        print(f'Loading the contact matrix for {ch}...')

        capture_string = ch_coord[ch]
        chrom, st, ed = get_coords(capture_string)

        assert st % interval == 0 and ed % interval == 0
        assert min_distance % interval == 0 and max_distance % interval == 0
        # Load matrix
        size = (ed - st) // resolution
        mat = get_matr(clr, capture_string, tns)

        if np.sum(mat) <= 0.001:
            print(
                f'Failed to load the contact matrix for {ch}! Check the input files.'
            )

        # Generate matrices
        batch_count = 0
        xs, ys, sub_mats = [], [], []
        print(f'Searching loops from {ch}:{st} to {ch}:{ed}...')
        current_coord = 0
        for x in range(st, ed, interval):
            for y in range(x, ed, interval):
                if min_distance <= y - x <= max_distance:
                    new_x, new_y = (x - st) // resolution, (y - st) // resolution
                    if new_x - window_size >= 0 and new_y - window_size >= 0 and new_x + window_size < size and new_y + window_size < size:
                        xs.append(x)
                        ys.append(y)
                        sub_mats.append(
                            normalize_mat(mat[new_x - window_size: new_x + window_size + 1,
                                          new_y - window_size: new_y + window_size + 1])
                        )
                if len(xs) == batch_size:
                    if batch_count % 500 == 0 and xs[-1] > current_coord:
                        print(f' Currently at coordinate: {ch}:{xs[-1]}')
                    yield batch_count, ch, xs, ys, np.array(sub_mats).reshape(
                        [len(xs), 2 * window_size + 1, 2 * window_size + 1, 1])
                    xs, ys, sub_mats = [], [], []
                    batch_count += 1

        if len(xs) > 0:
            yield batch_count, ch, xs, ys, np.array(sub_mats).reshape(
                [len(xs), 2 * window_size + 1, 2 * window_size + 1, 1])

# run pixel-wise imputation on input cooler
def identify_loops(run_name,
                   ch_coord, paths,
                   output, tns=None, threshold=0.99,
                   max_distance=1000000, min_distance=5000,
                   resolution=200, interval=1000, batch_size=10):
    """
    Identify the pixels that contains a loop
    """
    model = CNN_model(lr=lr)
    model.load_weights(f'{run_name}_model.h5')

    f_out = open(output, 'w')
    f_out.write(
        f'#chr anchor1 anchor2 loopLikelihood\n'
    )
    print(f'Beginning batch processing of size {batch_size}')
    for batch_count, ch, xs, ys, sub_mats in batch_load_test_mats_clr(tns=tns,
            ch_coord=ch_coord,
            max_distance=max_distance, min_distance=min_distance, clr_path=paths,
            resolution=resolution, interval=interval, batch_size=batch_size
    ):
        res = model.predict(sub_mats)
        for x, y, lb in zip(xs, ys, res):
            if lb[0] > threshold:
                f_out.write(
                    f'{ch}\t{x}\t{y}\t{lb[0]}\n'
                )

        if batch_count % 1000 == 0:
            print(f" batch count {batch_count}")
            f_out.flush()
    f_out.close()

# merge nearby loops by distance
# provides a quick merged output if not using the image processing merge
def merge_loops(input_file, output_file, p_thr=0.995, distance_thr=2001):
    """
    Merge the nearby loops from the results of identify_loops()
    """
    loop_idx = {}
    n_groups = {}
    f = open(input_file)
    for line in f:
        if line.startswith('#'):
            continue
        lst = line.strip().split()
        ch, p1, p2, p = lst[0], int(lst[1]), int(lst[2]), float(lst[3])
        if p < p_thr:
            continue
        if ch not in loop_idx:
            loop_idx[ch] = {
                (p1, p2): (0, p)
            }
            n_groups[ch] = 1
        else:
            found = False
            for (old_p1, old_p2) in loop_idx[ch]:
                if abs(old_p1 - p1) + abs(old_p2 - p2) < distance_thr:
                    loop_idx[ch][(p1, p2)] = (loop_idx[ch][(old_p1, old_p2)][0], p)
                    found = True
                    break
            if not found:
                loop_idx[ch][(p1, p2)] = (n_groups[ch], p)
                n_groups[ch] += 1

    f2 = open(output_file, 'w')
    f2.write(
        f'#chr anchor1 anchor2 loopLikelihood\n'
    )
    for ch in n_groups:
        n_loops = n_groups[ch]
        for i in range(n_loops):
            all_loops = [
                elm for elm in loop_idx[ch] if loop_idx[ch][elm][0] == i
            ]
            all_loops = sorted(all_loops, key=lambda x: loop_idx[ch][x][1], reverse=True)
            p1, p2 = all_loops[0]
            f2.write(f'{ch}\t{p1}\t{p2}\t{loop_idx[ch][all_loops[0]][1]}\n')
    f2.close()


# helper function to read in the input regions
# example input: {'region1':'chr1:1000-2000', 'region2':'chr2:4000-5000'}
def read_region_list_as_dict(regions_in):
    with open(regions_in) as f:
        regions_dict = {k: v for k, v in (line.split() for line in f)}

    return regions_dict


if __name__ == '__main__':
    ##
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    parser = argparse.ArgumentParser()
    parser.add_argument('cell_type', type=str, help='nametag for the sample')
    parser.add_argument('cooler_name', type=str, help='mcool file')
    parser.add_argument('regions_file', type=str, help='2-column file of regions and coordinates')

    parser.add_argument('-r', '--regions', type=str, default='all', help='comma-separated regions to analyze')
    parser.add_argument('-m', '--model_name', default='CHIRON_v0', type=str, help='base name of the model to impute')
    parser.add_argument('-o', '--out_dir', type=str, default='./', help='output directory')
    parser.add_argument('-b', '--batch_size', type=int, default=64)

    args = parser.parse_args()

    # required arguments
    model_name = args.model_name
    cell_type = args.cell_type
    clr_name = args.cooler_name
    region_list = args.regions_file
    batch_size = args.batch_size
    out_dir = args.out_dir

    # assume that the region list is in UCSC coordinates
    ch_coord_all = read_region_list_as_dict(region_list)

    if args.regions == 'all':
        ch_coord = ch_coord_all
    else:
        regions = [int(item) for item in args.regions.split(',')]
        ch_coord = {key: value for key, value in ch_coord_all.items() if key in regions}

    print(ch_coord)

    today = date.today()
    metric_dir = os.path.join(out_dir, f"{model_name}_{today}")

    if not os.path.exists(metric_dir):
        os.mkdir(metric_dir)

    identify_loops(
        model_name, ch_coord, clr_name, output=f'{metric_dir}/{cell_type}_{args.regions}_loops_raw.txt',
        threshold=0.99, max_distance=2000000, min_distance=6000,
        resolution=1000, interval=1000, batch_size=batch_size)

    # Merge the nearby loops by naive distance
    # Note that for Hong et al 2025, this is not the loop set used (loops are re-merged from the raw loops)
    merge_loops(
        input_file=f'{metric_dir}/{cell_type}_{args.regions}_loops_raw.txt', output_file=f'{metric_dir}/{cell_type}_{args.regions}_loops_merged.txt',
        p_thr=0.995, distance_thr=2001
    )
