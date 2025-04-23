import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
import matplotlib as mpl
import os
import matplotlib

pred_all = pd.read_csv('/mnt/md0/varshini/RCMC_LoopCaller/pred_thresholds_idli.txt', sep='\t', index_col=0)
pred_all_mus = pd.read_csv('/mnt/md0/varshini/RCMC_LoopCaller/pred_thresholds_mus.txt', sep='\t', index_col=0)
pred_all_pk = pd.read_csv('/mnt/md0/varshini/RCMC_LoopCaller/pred_thresholds_pk_full_chr45_filtered_newGM.txt', sep='\t', index_col=0)

#print(plt.rcParams.keys())

save_dir = '/mnt/md0/varshini/RCMC_LoopCaller/figs/'

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
plt.rcParams.update({'font.size': 16})

# trigger core fonts for PDF backend
plt.rcParams["pdf.use14corefonts"] = True
# trigger core fonts for PS backend
plt.rcParams["ps.useafm"] = True
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    "text.usetex": False})
plt.rcParams['svg.fonttype'] = 'none'

recall = pred_all.loc['TP',:].divide((pred_all.loc['TP',:] + pred_all.loc['FN',:]))
precision = pred_all.loc['TP', :].divide((pred_all.loc['TP', :] + pred_all.loc['FP', :]))

#print(recall, precision)
recall_pk = pred_all_pk.loc['TP', :].divide((pred_all_pk.loc['TP', :] + pred_all_pk.loc['FN', :]))
precision_pk = pred_all_pk.loc['TP', :].divide((pred_all_pk.loc['TP', :] + pred_all_pk.loc['FP', :]))

#pred_all_mus.fillna(1, inplace=True)
recall_mus = pred_all_mus.dropna(axis=1).loc['TP', :].divide((pred_all_mus.dropna(axis=1).loc['TP', :] + pred_all_mus.dropna(axis=1).loc['FN', :]))
precision_mus = pred_all_mus.dropna(axis=1).loc['TP', :].divide((pred_all_mus.dropna(axis=1).loc['TP', :] + pred_all_mus.dropna(axis=1).loc['FP', :]))

pr = pd.concat([recall_mus, precision_mus], axis=1)

recall_mus = pred_all_mus.loc['TP', :].divide((pred_all_mus.loc['TP', :] + pred_all_mus.loc['FN', :]))
recall_mus.fillna(recall_mus.dropna().iloc[0], inplace=True)
precision_mus = pred_all_mus.loc['TP', :].divide((pred_all_mus.loc['TP', :] + pred_all_mus.loc['FP', :]))
precision_mus.fillna(precision_mus.dropna().iloc[0], inplace=True)

print("AUCS by riemann:")
# calculate dx as recall[n+1] - recall[n]
recalli = recall.array[::-1]
precisioni = precision.array[::-1]
#print(recall, precision)
dx_idli = [recalli[i+1]-recalli[i] for i in range(0, len(recalli)-1)]
dx_mus = [recall_mus.array[i+1]-recall_mus.array[i] for i in range(0, len(recall_mus.array)-1)]
p=precisioni.fillna(np.max(precisioni))
#print(f"p: {p}")
auc_idli = np.sum([p[i]*dx_idli[i] for i in range(0, len(dx_idli))])
auc_mus = np.sum([precision_mus.array.fillna(0)[i]*dx_mus[i] for i in range(0, len(dx_mus))])
print(auc_idli)
print(auc_mus)


fig, ax = plt.subplots()
disp = metrics.PrecisionRecallDisplay(precision_mus.array, recall_mus.array)

disp2 = metrics.PrecisionRecallDisplay(precision_pk.array, recall_pk.array)
disp3 = metrics.PrecisionRecallDisplay(precision.array, recall.array)
#print(precision_pk)
disp.plot(ax=plt.gca(), linewidth=2, color='#BBBBBB')
#disp2.plot(ax=plt.gca(), linewidth=2, color='#33BBEE')
disp3.plot(ax=plt.gca(), linewidth=2, color='#0077BB')

plt.legend(['Mustache', 'IDLI'], frameon=False)
#plt.text(auprc_mus)
#print(auprc_mus)
#plt.ylim([0, 1])
#plt.title('Mustache')\
plt.xlim([0, 1])
plt.ylim([0, 1])
#plt.show()
plt.savefig(os.path.join(save_dir, 'pr_curve_chr45_filtered_1kb_GMonly_font16.svg'))
