import copy

import cv2
import numpy as np
import cooler
import os
import loop_utils_2 as loop_utils
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re
def process_mask(mask):
    # Discard any pixels with less than 4-connectivity
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    return labels, stats


def apply_gaussian_blur(image, k):
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (k, k), 0)
    return blurred_image


def dilate_region(region, kernel_size):
    # Perform dilation using a square kernel of the given size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_region = cv2.dilate(region, kernel, iterations=1)
    return dilated_region


def find_local_maxima(blob, dilated_blob, global_max):
    # Find the local maxima where blob pixel == dilated pixel
    local_maxima = (blob == dilated_blob)

    # Filter out local maxima that are less than 0.8 times the global max intensity
    local_maxima[blob < (global_max * intensity_threshold)] = False

    return local_maxima


def filter_blobs_by_coordinates(labels, coordinates):
    """
    Filters blobs by checking if any of the given coordinates fall within a specific blob.

    Args:
        labels (np.ndarray): Labeled mask with each blob assigned a unique label.
        coordinates (list of tuples): List of (x, y) coordinates to check within the blobs.

    Returns:
        set: A set of blob labels that contain any of the specified coordinates.
    """
    selected_labels = set()

    # Loop over the list of coordinates
    for y, x in coordinates:
        # Check the label of the blob at the given coordinate (x, y)
        label_at_point = labels[y, x]  # Remember: labels are indexed as [y, x]
        if label_at_point != 0:  # 0 is the background label
            selected_labels.add(label_at_point)
    return selected_labels


def plot_padded_region_and_maxima(padded_probs, local_maxima_coords):
    """
    Plot the padded region and mark the local maxima on the plot.

    Args:
        padded_probs (np.ndarray): The padded region with probability values.
        local_maxima_coords (np.ndarray): Array of local maxima coordinates relative to the padded region.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(padded_probs, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Probability")

    # Plot the local maxima as red dots
    for coord in local_maxima_coords:
        plt.scatter(coord[1], coord[0], color='red', s=50, label="Local Maxima")

    plt.title('Padded Region with Local Maxima')
    plt.show()

def process_blobs(image, mask, test_coordinates=None):
    # Process the mask to find connected blobs
    labels, stats = process_mask(mask)

    # Apply Gaussian blur to the image
    blurred_image = apply_gaussian_blur(image, k=3)

    merged_loops = []

    # If test coordinates are provided, filter blobs
    if test_coordinates:
        selected_blobs = filter_blobs_by_coordinates(labels, test_coordinates)
    else:
        # If no coordinates are provided, use all blobs
        selected_blobs = set(range(1, len(stats)))  # Exclude label 0 (background)

    # For each selected blob
    for label in selected_blobs:
        # Create the blob mask
        blob_mask = (labels == label).astype(np.uint8)

        # Find the bounding box of the blob with an m-pixel padding
        x, y, w, h = stats[label][:4]
        padded_region = blurred_image[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        region_not_blurred = image[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        # Perform dilation on the padded region
        dilated_region = dilate_region(padded_region, dilation_size)

        # Use the blob_mask to extract the region corresponding to the current blob
        blob = blurred_image[y:y + h, x:x + w] * blob_mask[y:y + h, x:x + w]
        dilated_blob = dilated_region[m:-m, m:-m]

        # Find the global maximum in the blob
        global_max = np.max(blob)

        # Find the local maxima in the blob
        local_maxima = find_local_maxima(blob, dilated_blob, global_max)


        # Add the local maxima coordinates to the merged loops list
        max_coords = np.column_stack(np.where(local_maxima))
        # plot_padded_region_and_maxima(region_not_blurred, max_coords + m)
        original_coords = max_coords + [y - m + 1, x - m + 1]
        for coord in original_coords:
            merged_loops.append(coord)

    return merged_loops

    # Save the merged loops to a file
    # with open("merged_loops.txt", "w") as f:
    #     for coords in merged_loops:
    #         for coord in coords:
    #             f.write(f"{coord[0]},{coord[1]}\n")


def process_blobs_probs(mask, blob_mask_df, test_coordinates=None):
    # Process the mask to find connected blobs
    labels, stats = process_mask(mask)

    merged_loops = []

    # If test coordinates are provided, filter blobs
    if test_coordinates:
        selected_blobs = filter_blobs_by_coordinates(labels, test_coordinates)
    else:
        # If no coordinates are provided, use all blobs
        selected_blobs = set(range(1, len(stats)))  # Exclude label 0 (background)
    # print(labels)
    # For each selected blob
    for label in selected_blobs:
        x, y, w, h = stats[label][:4]
        # Filter the blob_mask_df to get only the relevant pixels for this blob
        blob_mask = blob_mask_df[labels[blob_mask_df["anchor1"], blob_mask_df["anchor2"]] == label]
        # print(blob_mask)
        # Get the blob's coordinates and probabilities
        coords = blob_mask[["anchor1", "anchor2"]].values
        probs = blob_mask["loopLikelihood"].values

        # Calculate the median probability for padding
        m_prob = np.min(probs)
        # Find the bounding box of the blob with an m-pixel padding
        x, y, w, h = stats[label][:4]

        # Create a padded region with the median probability
        padded_probs = np.full((h + 2 * m, w + 2 * m), m_prob)
        # Fill the padded region with the actual probabilities from the blob
        for (y_coord, x_coord), prob in zip(coords, probs):
            padded_probs[(y_coord - y + m), (x_coord - x + m)] = prob

        # # Perform dilation on the padded region
        padded_probs = (padded_probs-min(padded_probs.flatten()))/(max(padded_probs.flatten())-min(padded_probs.flatten()))
        dilated_blob_probs = dilate_region(padded_probs, dilation_size)

        prob_mask = copy.deepcopy(padded_probs)
        prob_mask[prob_mask>m_prob] = 1
        prob_mask[prob_mask<=m_prob] = 0

        # blob = padded_probs[y:y + h, x:x + w] * prob_mask[y:y + h, x:x + w]
        # dilated_region_sub = dilated_blob_probs[m:-m, m:-m]

        # blob_probs = np.multiply(padded_probs,prob_mask)
        # plt.imshow(blob_probs)
        # plt.show()
        # Find the global maximum in the blob based on probabilities
        global_max = np.max(padded_probs)

        # Find the local maxima in the blob
        # plt.imshow(padded_probs)
        # plt.show()
        # plt.imshow(dilated_blob_probs)
        # plt.show()

        local_maxima = find_local_maxima(padded_probs, dilated_blob_probs, global_max)
        local_maxima = local_maxima * prob_mask
        # Add the local maxima coordinates to the merged loops list
        max_coords = np.column_stack(np.where(local_maxima))
        original_coords = max_coords + [y - m + 1, x - m + 1]
        print(original_coords)
        for coord in original_coords:
            merged_loops.append(coord)

        # Plot the padded region and the local maxima for this blob
        # plot_padded_region_and_maxima(padded_probs, max_coords)  # Adjust coordinates to padded region

    return merged_loops

def read_region_list_as_dict(regions_in):
    with open(regions_in) as f:
        regions_dict = {k: v for k, v in (line.split() for line in f)}

    return regions_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('region_list', type=str, help='2-column file of regions and coordinates')
    parser.add_argument('loop_dir', type=str, help='directory with all raw loop files to process')
    parser.add_argument('cooler_path', type=str, help='mcool path')
    parser.add_argument('outdir', type=str, help='where to save output')

    parser.add_argument('-r', '--regions', type=str, default='all',
                        help='comma-separated values for which regions to analyze')
    parser.add_argument('-d', '--dilation_size', type=int, default=7,
                        help='Dilation kernel (defines the max number of pixels between features)')
    parser.add_argument('-t', '--intensity_threshold', type=float, default=0.8,
                        help='value threshold (relative to the max) which is counted as a feature')
    parser.add_argument('-r', '--res', type=int, default=1000)
    parser.add_argument('-m', '--method', type=str, default='intensity',
                        help='options: intensity and probability (segment by image intensity or loop probability)')
    parser.add_argument('-o', '--outfile', type=str, default=None)

    args = parser.parse_args()

    dilation_size = args.dilation_size  # Dilation kernel size
    m = 5 % 2  # Padding for the rectangular region
    intensity_threshold = args.intensity_threshold  # Threshold for local maxima based on global maximum
    method = args.method
    clr_path = args.cooler_name
    loop_dir = args.loop_dir
    res = args.res
    outdir = args.outdir
    regions = args.regions
    region_list = read_region_list_as_dict(args.region_list)

    if args.regions == 'all':
        regions = [item for item in region_list.keys()]
    else:
        regions = [item for item in args.regions.split(',')]

    # for each cell type, load the cooler
    # iterate through all files in folder and select for "celltype_regions_raw.txt"
    # split the region substring and parse through
    for f in os.listdir(loop_dir):
        out_df = pd.DataFrame(columns=['chr1', 'anchor1', 'anchor2', 'loopLikelihood'])
        print(f)
        s = f.split('_')
        if 'merged' in s:
            continue

        ct=s[0]
        clr_name = f'{clr_path}::resolutions/1000'
        clr = cooler.Cooler(clr_name)
        loops_og = pd.read_csv(os.path.join(loop_dir, f), sep='\s+')

        for reg in regions:
            region_str = region_list[reg]
            print(region_str)

            chrom, start, end = loop_utils.get_coords(region_str)
            loops = loops_og[loops_og['anchor1'] >= start]
            loops = loops[loops['anchor2'] <= end]

            l = copy.deepcopy(loops['loopLikelihood'])
            image = loop_utils.get_matr(clr, region_str)
            mask = loop_utils.loops_to_matrix_bed(loops, start, end)

            loops['anchor1'] = (loops['anchor1'].to_numpy() - start) // 1000
            loops['anchor2'] = (loops['anchor2'].to_numpy() - start) // 1000

            # Process the blobs and save the maxima (filtered by test_coordinates)
            if method=='prob':
                out = process_blobs_probs(np.uint8(mask), loops, None)
            elif method == 'intensity':
                out = process_blobs(image, np.uint8(mask), None)
            else:
                print("Method not supported. Options are 'prob' and 'intensity' ")
                break

            out = pd.DataFrame(np.row_stack(out), columns=['anchor1', 'anchor2'])

            out = out.merge(loops.drop('#chr', axis=1), how='left', on=['anchor1', 'anchor2'])
            out['anchor1'] = out['anchor1'] * 1000 + start
            out['anchor2'] = out['anchor2'] * 1000 + start
            out['chr1'] = chrom
            out= out[['chr1', 'anchor1', 'anchor2', 'loopLikelihood']]
            out = out[out['loopLikelihood']>=0.99]
            out_df = pd.concat([out_df, out])

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.outfile is None:
            out_df.to_csv(os.path.join(loop_dir, f'{outdir}/{f.split(".")[1]}_merged_gaussian.txt'), index=False, sep='\t')
        else:
            out_df.to_csv(os.path.join(loop_dir, f'{outdir}/{args.outfile}'), index=False, sep='\t')

######################
    # process the mask - discard any pixels with less than 4-connectivity

    # group the remaining blobs using a blob detector where any contiguous pixels in the mask are designated a blob

    # apply a gaussian blur to the image

    # for each item in blobs:
        # designate the region as the rectangle with a m-pix pad around the widest portion of the mask
        # perform a m%2 dilation on the region
        # find where in the blob (not the padded region) the blob pixel = dilated pixel (identifies local maxima)
        # find the global maximum in the blob. discard any of the local maxima that are <0.8 the pixel intensity of the global max