import numpy as np
import utils
from skimage import measure
import copy
import matplotlib.pyplot as plt

# code translated from matlab to python from
# https://github.com/lhcgeneva/SPT/blob/master/Matlab/Kilfoil/fracshift.m
def fracshift(im, shiftx, shifty):
    # compute shift values
    # ipx and ipy are the integer portions of the centroid shift
    # shiftx and shifty are the fractional shifts
    ipx = int(np.fix(shiftx))
    ipy = int(np.fix(shifty))
    fpx = shiftx - ipx
    fpy = shifty - ipy

    if fpx < 0:
        fpx = fpx + 1
        ipx = ipx - 1
    if fpy < 0:
        fpy = fpy + 1
        ipy = ipy - 1

    # print(fpx, fpy)
    # print(ipx, ipy)
    image = copy.deepcopy(im)
    # circularly shift the image along each combo of x and y to interpolate the fractional image.
    imagex = np.roll(image, (ipy, ipx + 1), axis=(0, 1))
    imagey = np.roll(image, (ipy + 1, ipx), axis=(0, 1))
    imagexy = np.roll(image, (ipy + 1, ipx + 1), axis=(0, 1))
    image = np.roll(image, (ipy, ipx), axis=(0, 1))

    s0 = (1 - fpx) * (1 - fpy)
    sx = fpx * (1 - fpy)
    sy = (1 - fpx) * fpy
    sxy = fpx * fpy
    # each 1 px shift is weighted by the fractional pixels st the result is centered at the shifted centroid
    res = s0 * image + sx * imagex + sy * imagey + sxy*imagexy
    # res = ((1 - fpx )*(1 - fpy)).* image )+ (fpx *( 1 -fpy ) *imagex )+ (( 1 -fpx ) *fp y *imagey )+ (fpx * fpy * imagexy)
    return res

def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = ((w // 2), (h // 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask

def generous_circular_mask(image_shape, radius, buffer=1.0):

    h, w = image_shape
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0  # use fractional center
    Y, X = np.indices((h, w))
    dist = np.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)

    mask = dist <= (radius + buffer)
    return mask

# version of the fracshift function for analysis purposes.
# tracks how much the center shifts with each iteration.
def centering_fracshift_track_distances_masked(loop, win=2, iterations=10, plotting=False, mask_radius=5):

    ogc = (np.shape(loop)[0]) // 2
    loop_sub = loop[ogc - win:ogc + win+1, ogc - win:ogc + win+1]
    sz = np.shape(loop_sub)[0]
    c1 = sz// 2

    # inital center is the central coordinate of the loop (assumes prior max centering)
    ctr = (c1, c1)  # set init centroid coordinates

    print(f"Image Coord: {ctr}")
    print(f"Loop Matr Size: {np.shape(loop_sub)}")
    distances = []
    loop_mask = create_circular_mask(np.shape(loop_sub)[0], np.shape(loop_sub)[1], center=ctr, radius=mask_radius)
    #loop_mask = generous_circular_mask(np.shape(loop_sub), radius=mask_radius)

    shiftx=0
    shifty=0

    if plotting:
    # plot [col, row]
        utils.plot_simple(np.multiply(loop_sub, loop_mask))

    for i in range(iterations):

        # get weighted centroid, which is the first and last step of fracshift
        # this center is in [col, row] format

        # 1. find center (of the circshifted image) under the mask.
        if np.sum(np.sum(np.multiply(loop_sub, loop_mask)))==0:
            return (c1, c1), (0, 0), np.repeat(np.nan, iterations, axis=0), (0,0)
        ctr, shift = calc_center(np.multiply(loop_sub, loop_mask), (c1, c1), plotting=False)
        shiftx = shiftx + shift[0]  # column
        shifty = shifty + shift[1]  # row
        if i==0: # record initial centroid shift
            init_x=shiftx
            init_y=shifty

        # 2. create mask to use for the next iteration.
        #loop_mask = create_circular_mask(np.shape(loop_sub)[0], np.shape(loop_sub)[1], center=ctr, radius=7)
        if np.isnan(shift[0]) or np.isnan(shift[1]):
            return (c1, c1), (0, 0), np.repeat(np.nan, iterations, axis=0)

        # compute the distance of the shift
        dist = np.sqrt(shift[0]**2 + shift[1]**2)
        distances.append(dist)
        #print(shift)

        # 3. circshift the image based on the computed shift
        loop_sub = fracshift(loop_sub, -shift[0], -shift[1])

    #print(f"Matrix Coord After Fracshift: {ctr}")
    if plotting:
        # plot [col, row]
        utils.plot_simple(loop, (ogc + shiftx, ogc+shifty), ts="post fracshift")
        plt.figure()
        plt.plot(np.arange(1, len(distances)+1, 1), distances)
        plt.show()
    # return [row, col]
    final_ctr = (ogc + shiftx, ogc + shifty)
    return final_ctr, (shifty, shiftx), distances, (shifty-init_y, shiftx-init_x)


# the version without masking (be careful about window size!)
def centering_fracshift_track_distances(loop, win=2, iterations=10, plotting=False):
    ogc = (np.shape(loop)[0]) // 2
    loop_sub = loop[ogc - win:ogc + win+1, ogc - win:ogc + win+1]
    sz = np.shape(loop_sub)[0]
    c1 = sz// 2

    # inital center is the central coordinate of the loop (assumes prior max centering)
    ctr = (c1, c1)  # set init centroid coordinates

    print(f"Image Coord: {ctr}")
    print(f"Loop Matr Size: {np.shape(loop_sub)}")
    distances = []

    for i in range(iterations):
        # get weighted centroid, which is the first and last step of fracshift
        # this center is in [col, row] format
        ctr, shift = calc_center(loop_sub, ctr, plotting=plotting)

        if np.isnan(shift[0]) or np.isnan(shift[1]):
            return (c1, c1), (0, 0), np.repeat(np.nan, iterations, axis=0)

        # compute the distance of the shift
        dist = np.sqrt(shift[0]**2 + shift[1]**2)
        distances.append(dist)
        #print(shift)
        # basically, fracshift rolls over the image so that the new pixels represent parts of the prev.
        loop_sub = fracshift(loop_sub, shift[0], shift[1])

    #print(f"Matrix Coord After Fracshift: {ctr}")
    if plotting:
        # plot [col, row]
        utils.plot_simple(loop, (ctr[0] + ogc - win, ctr[1] + ogc - win), ts="post fracshift")
        plt.figure()
        plt.plot(np.arange(1, len(distances)+1, 1), distances)
        plt.show()
    # return [row, col]
    return ctr + (ogc, ogc), (ctr[1] - win, ctr[0] - win), distances


# Finds the center of a window using the fracshift method (iterative centroid)
def centering_fracshift(loop, win=2, iterations=10, plotting=False):
    # the original center (ogc) is the center coordinate (assumes prior max centering)
    ogc = (np.shape(loop)[0])//2

    # the center is only found in a window surrounding the loop center
    loop_sub = loop[ogc-win:ogc+win+1, ogc-win:ogc+win+1]
    sz = np.shape(loop_sub)[0]
    c1 = sz // 2

    # inital center is the central coordinate of the loop (assumes max centering)
    ctr = (c1, c1)  # set init centroid coordinates

    for i in range(iterations):
        # get weighted centroid, which is the first and last step of fracshift
        # this center is in [col, row] format
        ctr, shift = calc_center(loop_sub, ctr, plotting=False)

        # basically, fracshift rolls over the image so that the new pixels represent parts of the prev.
        loop_sub = fracshift(loop_sub, shift[0], shift[1])

    if plotting:
        # plot [col, row]
        utils.plot_simple(loop, (ctr[0]+ogc-win, ctr[1]+ogc-win), ts="post fracshift")
    # return [row, col]
    return ctr+(ogc,ogc), (ctr[1]-win, ctr[0]-win)

def calc_center(loop, ctr, plotting):

    # finds the center of the integrated intensity distribution
    extent = np.shape(loop)[0]
    pos = np.ones((extent, 1)) * np.arange(0, extent) # pixel positions, 0-indexed
    # m(i) = sum(sum(double(a(fix(yl(i)):fix(yh(i)),fix(xl(i)):fix(xh(i)))).*mask));
    mass = np.sum(np.sum(loop))
    xc = np.sum(np.sum(np.multiply(loop, pos))) / mass # the x-coordinate / column
    yc = np.sum(np.sum(np.multiply(loop, np.transpose(pos)))) / mass # the y-coordinate / row
    if plotting:
        utils.plot_simple(loop, (xc, yc))
    return (xc, yc), (xc-ctr[0], yc-ctr[1])

def loopsum(loop, p1, p2, win, weight_matr=None):
    # weighted sum
    if weight_matr is None:
        weight_matr = np.multiply(np.ones(np.shape(loop)), 0)
        weight_matr[p1-1:p1+2, p2-1:p1+2] = 0.00
        weight_matr[p1, p2] = 1
    loop = np.multiply(loop, weight_matr)

    loopsum = np.sum(loop[p1 - win:p1 + win + 1, p2 - win:p2 + win + 1])
    return loopsum

def calc_centroid(loop, ctr, plotting=False):
    # convert coordinates from image to matrix
    sz = np.shape(loop)[0]
    # print(f"old center: {ctr}")

    # find centroid within sub_loop (in matrix coord)
    new_ctr = measure.centroid(loop)

    shift = (new_ctr[0]-ctr[0], new_ctr[1]-ctr[1])
    # print(f"shift: {shift}")
    if plotting:
        utils.plot_simple(loop, (new_ctr[0], new_ctr[1]))
    return new_ctr, shift

def init_centering(loop, win=1, win2=1, plotting=False):
    # test all pix in 3x3 for 3x3 max
    sz = np.shape(loop)[0]
    c1 = sz // 2
    c2 = c1  # set centroid coordinates

    best_coord = (c1, c2)
    best_sum = loopsum(loop, c1, c2, win2)
    for i in range(-1 * win, win + 1):
        p1 = c1 + i
        for j in range(-1 * win, win + 1):
            p2 = c2 + j
            #print(f"searching {p1, p2}")

            # if p1==4 and p2 ==5:
            #     print(loopsum(loop, p1, p2, win))
            if loopsum(loop, p1, p2, win2) > best_sum:
                best_coord = (p1, p2)
                best_sum = loopsum(loop, p1, p2, win2)

            p2 = c2 - j
            if loopsum(loop, p1, p2, win2) > best_sum:
                best_coord = (p1, p2)
                best_sum = loopsum(loop, p1, p2, win2)
    if plotting:
        # [1 0] because converting from [row, col] (matrix indexing) to [col, row] (image indexing)
        utils.plot_simple(loop, (best_coord[1], best_coord[0]), ts = 'init center')
    # print(f"best coord: {best_coord}")
    return best_coord

def init_centering_upsample(loop, win=1, win2=1, plotting=False, ctr=None):
    # test all pix in 3x3 for 3x3 max
    sz = np.shape(loop)[0]

    if ctr is None:
        c1 = sz // 2
        c2 = c1  # set centroid coordinates
    else:
        (c1, c2) = ctr

    best_coord = (c1, c2)
    best_sum = loopsum(loop, c1, c2, win2)
    for i in range(-1 * win, win+1):
        p1 = c1 + i
        for j in range(-1 * win, win+1):
            p2 = c2 + j
            #print(f"searching {p1, p2}")

            # if p1==4 and p2 ==5:
            #     print(loopsum(loop, p1, p2, win))
            if loopsum(loop, p1, p2, win2) > best_sum:
                best_coord = (p1, p2)
                best_sum = loopsum(loop, p1, p2, win2)

            p2 = c2 - j
            if loopsum(loop, p1, p2, win2) > best_sum:
                best_coord = (p1, p2)
                best_sum = loopsum(loop, p1, p2, win2)
    if plotting:
        # [1 0] because converting from [row, col] (matrix indexing) to [col, row] (image indexing)
        utils.plot_simple(loop, (best_coord[1], best_coord[0]), ts = 'init center')
    #print(f"best coord: {best_coord}")
    return best_coord

# input: centered coordinate at res A, which is some integer factor of resolution B
## TODO: go back into setup.py if you hit the edge of the frame
def upsample_loop(cooler_mats, loop, curr_ctr, curr_res, target_res, capture_string, window=9, plotting=False, mask_radius=0):
    # chrom, st, ed = utils.get_coords(capture_string)
    res_factor = curr_res // target_res
    print(f"UPSAMPLING FROM {curr_res} to {target_res} res factor {res_factor}")

    div, rem = divmod(curr_res, target_res)
    assert rem == 0, "Target resolution must be factor of current resolution"

    # get new_ctr to res_factor. since the ctr pixel corresponds to res_factor number of pixels, subtract to
    # "average" the new loop center
    ctr_res = [int(np.round(c * res_factor)) - res_factor // 2  for c in curr_ctr]
    (c1, c2) = ctr_res
    ctr = [(c * res_factor) - res_factor // 2  for c in curr_ctr]
    # get matr at curr res
    new_res_mat = cooler_mats[target_res]

    # print(curr_ctr)
    # print(c1, c2)
    # print(np.shape(new_res_mat))

    # get sub mat
    hw = window // 2
    sub_matr = utils.get_sub_matr(new_res_mat, c1, c2, hw)

    #recenter
    init_center_upsamp = init_centering_upsample(sub_matr, win=res_factor//2, plotting=plotting)
    # [row, col] indexing
    ctr = (ctr[0] + (init_center_upsamp[0] - hw), ctr[1] + (init_center_upsamp[1] - hw))

    if mask_radius > 0: # for using masks, need to make sure the window is really centered on the max within the mask
        init_center_upsamp = init_centering_upsample(sub_matr, win=mask_radius, plotting=plotting,
                                                     ctr=(init_center_upsamp[0], init_center_upsamp[1]))
        ctr_res = (ctr_res[0] + (init_center_upsamp[0] - hw), ctr_res[1] + (init_center_upsamp[1] - hw))
        (c1, c2) = ctr_res
    sub_matr = utils.get_sub_matr(new_res_mat, c1, c2, hw)
    ## TESTING CODE
    if plotting:
    #     og_ctr = [int(np.round(c)) for c in curr_ctr]
    #     curr_res_mat = cooler_mats[curr_res]
    #     sub_matr_og = utils.get_sub_matr(curr_res_mat, og_ctr[0], og_ctr[1], hw)
    #     utils.plot_simple(sub_matr_og, ts=str(curr_res))
          utils.plot_simple(sub_matr, ts = str(target_res))
    #     # bin sub_matr to make sure its right
    #
    #     sub_matr_2 = utils.get_sub_matr(new_res_mat, c1, c2, hw*res_factor)
    #     binned_matr = utils.bin_matr(sub_matr_2[1:, 1:], res_factor)
    #     utils.plot_simple(binned_matr)

    return ctr, (c1, c2), sub_matr
