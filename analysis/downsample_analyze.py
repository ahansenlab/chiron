import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from fractions import Fraction

win='win3'
NUM_CHR_READS = 604258793
#NUM_CHR_READS = 591379253
CHR='chr6'
save_dir = '/mnt/md0/varshini/RCMC_LoopCaller/figs_150425/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
mpl.font_manager.fontManager.addfont("/home/varshini/anaconda3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/afm/phvr8a.afm")
mpl.font_manager.fontManager.addfont("/home/varshini/.virtualenvs/Peakachu/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts/Helvetica.afm")
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = True

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.linewidth'] = 0.25
plt.rcParams['xtick.major.width'] = 0.25
plt.rcParams['ytick.major.width'] = 0.25
plt.rcParams.update({'font.size': 14})

# trigger core fonts for PDF backend
plt.rcParams["pdf.use14corefonts"] = True
# trigger core fonts for PS backend
plt.rcParams["ps.useafm"] = True
plt.rcParams.update({
    "text.usetex": False})
plt.rcParams['svg.fonttype'] = 'none'

wins = ['win3']

bad_all_wins = []
legs=[]
leg3=[]
for win in wins:
    base_dir = f'/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/fracshift_downsample_v2/{win}'

    if win == 'win5.400' or win == 'win5v1':
         base_calls_og = pd.read_csv(os.path.join(base_dir, 'GM12878_loops_correction.win5_1.txt'), sep='\t')
    else:
        base_calls_og = pd.read_csv(os.path.join(base_dir, 'GM12878_loops_correction_1.txt'), sep='\t')
        #base_calls_og = pd.read_csv(os.path.join(base_dir, 'GM12878_loops_correction.win7_1.txt'), sep='\t')
    for chr in ['all']:
        fig_cdf, ax_cdf = plt.subplots(figsize=(10, 8))

        #base_calls = base_calls_og[base_calls_og['chr1']==chr].reset_index(drop=True).drop_duplicates(subset=['start1', 'end1'],)
        base_calls= base_calls_og[(base_calls_og['chr1'] == CHR)].reset_index(drop=True)
        #base_calls = base_calls_og
        print(base_calls)
        dis_all = []
        leg = []
        leg1 = []
        bad_all = []
        std_all = []
        cmap = np.array(sns.color_palette("ch:start=.2,rot=-.3", n_colors=8))

        ct=0
        for f in os.listdir(base_dir):
            if f == 'GM12878_092324.txt':
                continue
            if f == 'GM12878_loops_correction.win5_1.txt':
                continue
            #base_calls = pd.read_csv(f'/mnt/md0/varshini/RCMC_LoopCaller/loopcalls/fracshift_downsample/win3/{f.replace(".win5", "")}', sep='\t')
            print(f)
            frac = f.split('_')[3][:-4]
            curr_calls = pd.read_csv(os.path.join(base_dir,f), sep='\t')

            curr_calls = curr_calls[(curr_calls['chr1']==CHR)].reset_index(drop=True)
            print(len(curr_calls))
            start_dis = (curr_calls['start1'] - base_calls['start1']).dropna()
            end_dis = (curr_calls['end1'] - base_calls['end1']).dropna()

            bad_ind = curr_calls[curr_calls['converge']==False].index

            #start_dis.drop(bad_ind, axis=0, inplace=True)
            #end_dis.drop(bad_ind, axis=0, inplace=True)

            dis = np.sqrt(np.square(start_dis.to_numpy()) + np.square(end_dis.to_numpy()))
            #dis = end_dis.to_numpy()
            dis_all.append(dis)
            print(np.median(dis))
            print(np.std(dis))
            std_all.append(np.std(dis))
            bad_all.append(len(bad_ind))
            leg.append(frac)
            leg3.append(f)

            count1, bins_count1 = np.histogram(dis, bins=50)
            #print(count1)
            cdf1 = np.cumsum(count1 / sum(count1))
            #ax_cdf.plot(bins_count1[1:], cdf1, linewidth=2)
            leg1.append(float(frac))
            # if frac != "1":
            #     leg1.append(float(frac))

        ind = np.argsort(leg1)[::-1]
        print(leg1)
        ct=0
        for ct, i in enumerate(ind[1:]):
            #if divmod(ct, 2)[1]==0 or divmod(ct, 2)[1]!=0:
            if ct in [0, 1, 2, 3, 5, 7, 9, 11]:
                count1, bins_count1 = np.histogram(dis_all[i], bins=100)
               # ax_cdf.plot(bins_count1[1:], count1, linewidth=2.5, color=cmap[ct], label=str(Fraction(leg1[i])))
                ax_cdf.plot(bins_count1[1:], count1, linewidth=2.5, color=cmap[ct],
                            label=f"{np.format_float_positional((leg1[i]*NUM_CHR_READS)/1e6, 3, unique=False, trim='-', fractional=False)}")
        handles, labels = plt.gca().get_legend_handles_labels()
        print(handles, labels)
        plt.xlabel('distance from 100% reads center (bp)')
        plt.ylabel('number of loops')
        ax_cdf.spines['bottom'].set_color('k')
        ax_cdf.spines['left'].set_color('k')
        plt.xlim([200, 1400])
        #plt.xlim([0, 2500])
        plt.ylim([0, 80])
        plt.legend(frameon=False, title='number of reads (million)', alignment='left')
        #plt.show()
        plt.savefig(os.path.join(save_dir,'distance_vs_reads_sub.svg'), transparent=None, dpi=300)
        fig, ax = plt.subplots(figsize=(10, 8))
        # plot standard deviation vs # of reads
        plt.scatter(leg1, std_all)
        plt.xlabel('Number of of Reads')
        plt.ylabel('Standard Deviation from 100%')
        plt.ylim([300, 650])
        plt.show()


    #plt.savefig(os.path.join(save_dir, f'hist_compare_{win}.svg'), transparent=None, dpi=300)
    bad_all_wins.append(bad_all)
    legs.append(leg)
print(bad_all_wins)
converge_colors = ['#0077BB', '#33BBEE', '#009988']
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(len(wins)):
    print(i)
    #ax.scatter([float(l) for l in legs[i]], bad_all_wins[i], color=converge_colors[i], s=70, label=[wins[i]])
    ax.plot(np.sort([(float(l)*NUM_CHR_READS)/1e6 for l in legs[i]]), (1640-np.sort(bad_all_wins[i])[::-1])/1640, '--o', color=converge_colors[i], label=wins[i])
plt.axvline(8118321/1e6, color='#CC3311', linestyle='--')
plt.xlabel('number of reads (million)')
plt.ylabel('fraction of converged loop centers')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')

plt.legend(frameon=False)
#plt.show()
plt.savefig(os.path.join(save_dir, f'converged_loops_all.svg'), transparent=None, dpi=300)

fig, ax = plt.subplots(figsize=(8, 4))
plt.violinplot(dis_all)
ax.set_xticks(np.arange(1, len(leg)+1, 1), labels=leg, fontsize=8)
plt.title(chr)
plt.show()

    # fig, ax = plt.subplots(figsize=(8, 4))
    # plt.bar(leg, bad_all)
    # ax.set_xticks(np.arange(0, len(leg), 1), labels=leg)
    # plt.title(chr)
    # plt.show()
monomer_types_per_chrom = np.concatenate((np.zeros(50),
np.tile(np.concatenate((np.array([2,2]), np.zeros(18))), 4),
np.concatenate((np.ones(2), np.zeros(18))), np.zeros(50))).astype(int)
