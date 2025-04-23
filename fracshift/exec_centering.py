'''
exec_centering.py
written by Varshini Ramanathan (varsh@mit.edu)
purpose: center loop calls to sub-binsize precision using the fracshift algorithm
'''

import setup
import utils
import centering
import pandas as pd
import numpy as np
import argparse
import logging

def read_region_list_as_dict(regions_in):
    with open(regions_in) as f:
        regions_dict = {k: v for k, v in (line.split() for line in f)}

    return regions_dict

def main(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('region_list', type=str)
    parser.add_argument('loop_path', type=str)
    parser.add_argument('cooler_path', type=str)
    parser.add_argument('outdir', type=str)

    parser.add_argument('-r', '--regions', type=str, default='all')
    parser.add_argument('-f', '--fracshift_mask', type=int, default=3)
    parser.add_argument('-m', '--max_window', type=int, default=1)
    parser.add_argument('-l', '--logfile', type=str, default='logfile')
    parser.add_argument('-i', '--init_res', type=int, default=800)
    parser.add_argument('-t', '--target_res', type=int, default=200)
    parser.add_argument('-v', '--variable_res', action='store_true')
    parser.add_argument('-n', '--iterations', type=int, default=20)
    parser.add_argument('-o', '--outfile', type=str, default='fracshift_loops')
    args = parser.parse_args(raw_args)

    # read region_list (tab separated) in as a dictionary
    region_list = read_region_list_as_dict(args.region_list)

    loop_fname = args.loop_path
    fracshift_mask = args.fracshift_mask
    max_window = args.max_window
    logfile = args.logfile
    target_res = args.target_res
    clr_path = args.cooler_path
    outdir = args.outdir
    iter=args.iterations
    outfile=args.outfile

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{logfile}.log", format="%(levelname)s: %(message)s", level=logging.INFO)

    logger.info(f"PARAMETERS: \n initial resolution: {args.init_res} \n target resolution: {target_res} \n "
                f"intial shift window: {max_window} \n fracshift mask: {fracshift_mask}\n")

    if args.regions == 'all':
        regions = [item for item in region_list.keys()]
    else:
        regions = [item for item in args.regions.split(',')]
    logger.info(f"regions: {regions}")

    corrected_calls = pd.DataFrame(columns=['chr1', 'start1', 'end1', 'converge'])

    # make resolution matching dictionary ** TODO change to accept cl arguments
    if args.variable_res:  # set initial resolution based on genomic distance
        res_list0 = [args.init_res, args.init_res * 2, args.init_res * 4, args.init_res * 8]
        logger.info(res_list0)
        cutoffs = {'res': res_list0,
                   'cutoffs': [500, 1000, 2000, np.inf]}
        targets = res_list0[::-1]
        targets.append(target_res)
    else:
        cutoffs = {'res': [args.init_res], 'cutoffs': [np.inf]}
        targets = [args.init_res, target_res]

    div, rem = divmod(args.init_res, target_res)
    assert rem == 0, "Target resolution must be factor of all resolutions"

    # these are all the possible resolutions you could subsample into. They have to be 2-multiples of each other
    targets.sort()

    ## TODO assert that target resolutions are all integer factors of the cutoff resolutions

    ## TODO dynamically compute these based on mask radius. but these numbers should be safe for any
    # reasonable loop radii/binsizes.
    full_win = 19  ### MUST BE ODD ####
    WINFRAC = 9
    for reg in regions:
        capture_string = region_list[reg]
        chrom, st, ed = utils.get_coords(capture_string)

        # read loops in (ASSUME HEADER). if annotation resolution isn't saved, create it via distance
        loops_in = pd.read_csv(loop_fname, sep='\s+')
        loops_in.columns.values[0]='chr1'
        loops_in.columns.values[1] = 'start1'
        loops_in.columns.values[2] = 'end1'
        loops_in = loops_in[loops_in['chr1'] == chrom].drop_duplicates()
        loops = setup.format_loopcalls(loops_in, capture_string, cutoffs)
        loops.reset_index(drop=True, inplace=True)

        # make res_list from cutoffs and targets
        res_list = setup.make_res_list(cutoffs, targets)

        # load in the cooler matrices at all res in the cutoff matr / target set
        logger.info("Setting up RCMC matrices...")
        cooler_mats = setup.get_cooler_mats(clr_path, capture_string, res_list)

        # subset out the loops (memory intensive step)
        loop_mats = setup.get_loop_mats(cooler_mats, loops, capture_string, window=full_win)
        logger.info(f"Processing {len(loop_mats)} loops...")

        # what loops in your set you want to plot
        test_range = np.arange(0, len(loop_mats), 1000)

        # The parameters that have the most effect on the deviation of the point from the original center are
        # the initial shift window and mask radius. optimize these using test_fracshift.py
        distances = []
        for idx, loop in loops.iterrows():
            res = loop['anno_res']

            # The initial center of the loop
            ctr = (np.round((loop['start1'] - st) / res), np.round((loop['end1'] - st) / res))
            bp_ctr = (ctr[0] * res + st, ctr[1] * res + st)
            logger.info(f"original center: {ctr} corresponds to {bp_ctr}")
            curr_loop = loop_mats[idx]

            # floor division relies on the shape of the loop being odd
            half_sz = np.shape(curr_loop)[0] // 2

            if idx in test_range:
                plotting = True
            else:
                plotting = False

            ## STEP 1: SET THE CENTER TO THE LOCAL MAXIMUM
            init_center = centering.init_centering(curr_loop, win=max_window, plotting=plotting)

            # now the init ctr is in [row, col]
            init_ctr_coord = (ctr[0] + (init_center[0] - half_sz), ctr[1] + (init_center[1] - half_sz))
            logger.info(f"after init centering: {init_ctr_coord}")

            ## STEP 2: UPSAMPLE LOOP TO INTENDED RESOLUTION
            # center still in [row, col]
            ctr_flt, ctr, curr_loop = centering.upsample_loop(cooler_mats, loop, init_ctr_coord, res, target_res,
                                                              capture_string, window=full_win,
                                                              plotting=plotting)
            resamp = res // target_res
            res = target_res

            ## STEP 3: GET A FRACTIONAL CENTER WITH FRACSHIFT
            # center is computed in [col, row] but returned in [row, col]
            # print(f"curr loop: {np.shape(curr_loop)}") # debug statement not even useful for logger
            cctr, shift, dist, fshift = centering.centering_fracshift_track_distances_masked(curr_loop, win=WINFRAC,
                                                                                        plotting=False,
                                                                                        iterations=iter,
                                                                                        mask_radius=fracshift_mask)

            # record distances
            distances.append(dist)
            # [row, col]
            new_ctr = (ctr[0] + shift[0], ctr[1] + shift[1])
            logger.info(f"Before Fracshift: {ctr}, After Fracshift: {new_ctr}, Shift = {shift}")

            # REFORMAT CENTER FOR PLOTTING
            sz = np.shape(curr_loop)[0]
            c1 = (sz) // 2
            plt_ctr = (c1 + shift[0], c1 + shift[1])  # update the center for display
            if idx in test_range:
                # [col, row]
                utils.plot_simple(curr_loop, (plt_ctr[1], plt_ctr[0]), ts=f'test {resamp}', sd=f"{outdir}/loop_{idx}_{loop['start1']}_{loop['end1']}.jpg")

            # Now that you have your loop center in pixels, convert to genomic coordinates and save
            new_ctr_genecoord = (np.round(res * new_ctr[0] + st), np.round(res * new_ctr[1] + st))
            logger.info(f"New Center in BP: {new_ctr_genecoord}")

            if np.isnan(np.array(dist).all()):
                convergence = False
            else:
                # deviation should be within a 10th of a pixel
                # honestly, a better empiric for convergence should be used
                if np.max(dist[-3:]) > 0.1:
                    convergence = False
                else:
                    convergence = True
            new_row = [loop['chr1'], new_ctr_genecoord[0], new_ctr_genecoord[1], convergence]
            if plotting:
                final_ctr = (int(np.round((new_ctr_genecoord[0] - st) // target_res)),
                             int(np.round(new_ctr_genecoord[1] - st) // target_res))
                new_res_mat = cooler_mats[target_res]
                sub_matr = utils.get_sub_matr(new_res_mat, int(final_ctr[0]), int(final_ctr[1]), half_sz)
                utils.plot_simple(sub_matr)
            corrected_calls.loc[len(corrected_calls)] = new_row

    corrected_calls.to_csv(f"{outdir}/{outfile}.tsv", sep='\t', index=False)

if __name__ == '__main__':
    main()
