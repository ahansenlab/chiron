import setup
import utils
import centering
import pandas as pd
import os
import numpy as np

# Purpose: analyze the fracshift convergence as a function of read depth
# Determine the minimum read depth necessary to find 200 bp centers

import setup
import utils
import centering_archival
import pandas as pd
import os
import numpy as np

print(os.getcwd())

##################################################
# DEFINITIONS

loop_fname = '/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/Annotations_092324/GM12878_092324.txt'
ct = 'GM12878'

regions = {
        'region1': 'chr6:25111000-28622000', # in training set
        'region2': 'chr19:36027000-39757000', # in training set
        'region3': 'chr5:157000000-160150000', # Done (in held out set)
        'region4': 'chr1:207626000-210339000', # in training set
        'region5': 'chr4:61369000-64435000', # Done (in held out set)
        # 'region6': 'chr8:124988000-129796000',
        # 'region7': 'chr6:29678000-32257000',
        # 'region8': 'chrX:47081000-49443000',
        # 'region9': 'chr1:23728000-240532000',
        # 'region10': 'chr7:10480000-13507000',
        # 'region11': 'chr8:62918000-66567000',
        # 'region12': 'chr4:181289000-184015000',
        # 'region13': 'chr3:118624000-121534000',
        # 'region14': 'chr9:106603000-109910000'
    }

clr_path_base = '/mnt/coldstorage/clarice/RCMC_analyses/downsampled/GM12878'
clr_path_base = '/mnt/md0/varshini/RCMC_LoopCaller/data2/'
# read all mcools and split by _ to extract read fraction
for clr_path in os.listdir(clr_path_base):
    if not clr_path.endswith('mcool'):
        continue
    else:
        frac = clr_path.split('_')[2]
    corrected_calls = pd.DataFrame(columns=['chr1', 'start1', 'end1', 'converge'])
    for reg in regions:
        # make resolution matching dictionary ** TODO change to accept cl arguments
        # for targets and cutoffs
        # specify final target resolution [or until convergence] and number of iterations
        capture_string = regions[reg]
        # You probably want to change the utils function to make it convenient for you
        cutoffs = utils.make_default_cutoffs()

        # Here are some examples of cutoffs that I used
        # low depth example
        cutoffs = {'res': [1600, 3200, 6400], 'cutoffs': [50, 1000, np.inf]}
        # for lower depth
        cutoffs = {'res': [800, 1600, 3200, 6400], 'cutoffs': [50, 1000, 2000, np.inf]}
        #for higher depth
        cutoffs = {'res': [400, 800, 1600, 3200, 6400], 'cutoffs': [150, 500, 1000, 2000, np.inf]}
        cutoffs = {'res': [400, 800, 1600], 'cutoffs': [150, 500, np.inf]}
        cutoffs = {'res': [800], 'cutoffs': [np.inf]}
        # these are all the possible resolutions you could subsample into. They have to be 2-multiples of each other
        targets = [6400, 3200, 1600, 800, 400, 200]
        targets.sort()

        ## TODO assert that target resolutions are all integer factors of the cutoff resolutions

        full_win= 19 ### MUST BE ODD ####
        num_resamp = 2 # how many times to upsample

        #################################################

        chrom, st, ed = utils.get_coords(capture_string)

        # read loops in (ASSUME NO HEADER). if annotation resolution isn't saved, create it via distance
        # IMPORTANT: my weird convention is that I store point calls in bed format with the coords as 'start1', 'end1'
        loops_in = pd.read_csv(loop_fname, sep='\s+', names=['chr1', 'start1', 'end1'])
        loops_in = loops_in[loops_in['chr1']==chrom].drop_duplicates()
        loops = setup.format_loopcalls(loops_in, capture_string, cutoffs)
        loops.reset_index(drop=True, inplace=True)
        print(loops)
        # make res_list from cutoffs and targets
        res_list = setup.make_res_list(cutoffs, targets)

        # load in the cooler matrices at all res in the cutoff matr / target set
        cooler_mats = setup.get_cooler_mats(os.path.join(clr_path_base, clr_path), capture_string, res_list)

        # subset out the loops (memory intensive step)
        loop_mats = setup.get_loop_mats(cooler_mats, loops, capture_string, window=full_win)
        print(f"loops: {len(loop_mats)}")

        # what loops in your set you want to plot
        test_range = np.arange(0, len(loop_mats), 50)
        test_range= []
        # initialize output


        # The parameters that have the most effect on the deviation of the point from the original center are
        # the initial shift window and centroid window (any parameter called "win"). By default, I set it
        # to half the resampling factor, which ensures that it doesn't deviate from the original centroid
        # found in step 1.
        distances = []
        for idx, loop in loops.iterrows():
            print(idx)

            res = loop['anno_res']

            # The initial center of the loop
            ctr = ((loop['start1'] - st) // res, (loop['end1'] - st)//res)

            curr_loop = loop_mats[idx]
            #print(np.shape(curr_loop))

            # floor division relies on the shape of the loop being odd
            half_sz = np.shape(curr_loop)[0] // 2
            target_res = 200

            if idx in test_range:
                plotting=True
            else:
                plotting=False

            ## STEP 1: SET THE CENTER TO THE LOCAL MAXIMUM
            # allow 1kb movement You may want to make this smaller
            max_win = int(np.ceil(1 / int(np.ceil((res / 1000)))))
            init_center = centering.init_centering(curr_loop, win=max_win, plotting=plotting)
            # now the init ctr is in [row, col]
            init_ctr_coord = (ctr[0] + (init_center[0]-half_sz), ctr[1] + (init_center[1]-half_sz))

            # print(ctr)
            # print(init_ctr_coord)

            ## STEP 2: UPSAMPLE LOOP TO INTENDED RESOLUTION
            # center still in [row, col]
            ctr_flt, ctr, curr_loop = centering.upsample_loop(cooler_mats, loop, init_ctr_coord, res, target_res,
                                                              capture_string, window=full_win)
            resamp = res // target_res
            # print(ctr)
            # print(ctr_flt)
            res = target_res
            print(resamp)

            #init_ctr_upsamp_coord = (ctr[0] + (init_center_upsamp[0] - half_sz), ctr[1] + (init_center_upsamp[1] - half_sz))
            ## STEP 3: GET A FRACTIONAL CENTER WITH WEIGHTED CENTROID OR FRACSHIFT
            # To test fracshift, you should replace it with centering.centering_fracshift
            # center is computed in [col, row] but returned in [row, col]
            print(f"curr loop: {np.shape(curr_loop)}")
            # cctr, shift, dist = centering.centering_fracshift_track_distances(curr_loop, win=5, plotting=plotting, iterations=20)
            cctr, shift, dist, fshift = centering.centering_fracshift_track_distances_masked(curr_loop, win=9,
                                                                                        plotting=False,
                                                                                        iterations=20,
                                                                                        mask_radius=3)
            # record distances
            distances.append(dist)
            # [row, col]
            new_ctr = (ctr[0] + shift[0], ctr[1] + shift[1])
            print(f"shift: {shift}")
            print(f"Old Res Center: {init_ctr_coord}, New Res Center: {new_ctr}, Shift = {shift}")

            # REFORMAT CENTER FOR PLOTTING
            sz = np.shape(curr_loop)[0]
            c1 = (sz - 1) // 2
            plt_ctr = (c1 + shift[0], c1 + shift[1]) # update the center for display
            if idx in test_range:
                # [col, row]
                utils.plot_simple(curr_loop, (plt_ctr[1],plt_ctr[0]), ts=f'test {resamp}')
                # plt.figure()
                # plt.plot(np.arange(1, full_win+1, 1), np.mean(curr_loop, axis=0))
                # plt.title('Signal Distribution')
                # plt.show()

            ## TESTING ###
            # init_center = centering.init_centering(curr_loop)
            # init_shift = (init_center[0] - half_sz, init_center[1] - half_sz)
            # shift = centering.init_centering_moment(curr_loop, init_center, win=shift_win, plotting=False)
            # new_ctr = (ctr[0] + init_shift[0] + shift[0], ctr[1] + init_shift[1] + shift[1])
            ###

            # Now that you have your loop center in pixels, convert to genomic coordinates and save
            new_ctr_genecoord = (np.round(res * new_ctr[0] + st), np.round(res * new_ctr[1] + st))
            print(new_ctr_genecoord)
            print(dist)
            if np.isnan(np.array(dist).all()):
                convergence = False
                print("hi1")
            else:
                if np.max(dist[-3:]) > 0.1:
                    print("hi")
                    convergence = False
                else:
                    convergence = True
            new_row = [loop['chr1'], new_ctr_genecoord[0], new_ctr_genecoord[1], convergence]
            if plotting:
                diff_bp = (new_ctr_genecoord[0] - loop['start1'], new_ctr_genecoord[1] - loop['end1'])
                print(f"diff: {diff_bp}")
                diff_pix = (diff_bp[0] / loop['anno_res'], diff_bp[1] / loop['anno_res'])

                final_ctr = (int(np.round((new_ctr_genecoord[0] - st) // target_res)), int(np.round(new_ctr_genecoord[1] - st) // target_res))

                # diff_pix = (new_ctr_genecoord[1]/loop['anno_res']-loop['end1']/loop['anno_res'],
                #                                    new_ctr_genecoord[0]/loop['anno_res']-loop['start1']/loop['anno_res'])
                new_pix = (diff_pix[1] + (sz - 1) // 2, diff_pix[0] + (sz - 1) // 2)
                print(new_pix)
                #utils.plot_simple(loop_mats[idx], new_pix)

                new_res_mat = cooler_mats[target_res]
                sub_matr = utils.get_sub_matr(new_res_mat, int(final_ctr[0]), int(final_ctr[1]), half_sz)
                utils.plot_simple(sub_matr)
            corrected_calls.loc[len(corrected_calls)] = new_row
        np.savetxt(f"/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/fracshift_downsample_v2/dis/{reg}_{frac}_distances_win3.txt", distances)
        #print(np.array(distances))

    corrected_calls.to_csv(f'/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/fracshift_downsample_v2/win3/{ct}_loops_correction_{frac}.txt', sep='\t', index=False)
    ## TODO quit if the initial centering has some flag ex. too hard to center
