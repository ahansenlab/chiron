Loop calling and centering with CHIRON+fracshift

####

Loop calling with CHIRON:

CHIRON is a CNN that was built to classify loops in high-depth RCMC. Find below instructions on training and running the model (note that most users can jump right to imputation and loop merging):

1. Data fetching and pre-processing scripts (preprocessing/)
get_pretraining_data.py
get_neg_pretraining_data.py

get_training_data.py
get_neg_training_data.py

The negative data scripts rely on the corresponding positive data scripts, but the pretraining and fine-tuning data can be generated independently.

2. Training and imputation: (chiron/) 

a. Pre-training:model_pretrain.py
b. Fine-tuning: model_finetune.py
c. Imputation: imputation_by_region.py
d. Loop merging: merge_loops_gaussian.py

Until the fine-tuning step, all the scripts contain hard-coded paths and should be modified for each user.The fine-tuning, imputation, and loop merging can be run by specifying parameters in the command line flags (see -h flag for help).

3. fracshift (fracshift/):

After calling loops, there is an optional step to localize the loop centers to a theoretical "base pair" resolution (in practice, it will be limited by your read depth to about 0.1x the bin size at which the matrix is "filled in" i.e. there is sufficient signal in most of the bins). The fracshift application runs a user-facing script that runs through all the loops and applies fracshift; this script runs a few custom modules that implement all the functions.

User-facing scripts:
a. fracshift_exec_demo.py: shell that runs a demo of exec_centering.py
b. exec_centering.py: main script that reads all the loops and applies fracshift.

Modules:
c. centering.py: contains all centering/upsampling functions
d. setup.py
e. utils.py

Required inputs:
- 'region_list': a 2-column file with names for the regions and the UCSC genome coordinates for each
- 'loop_path': path to the loopcalls which should be tab-delimited point calls, as in the output of CHIRON.

###

DEMO

Example pre-training:
python /home/varsh/LoopCaller/code/model_pretrain.py 3 "$n" /pool001/varsh/loop_pretraining_v2 -l 0.001 -e 100 -b 64

Example fine-tuning:
python /home/varsh/LoopCaller/code/model_finetune.py pre_hp_4 ftr UNFREEZE -u -6 -d 0.3 -b 512 -e 50 -l 0.001 -g True

Example imputation:
python /home/varsh/LoopCaller/code/imputation_by_region_general.py pre_hp_4_ftr \
    GM12878 GM12878_merged_realigned.50.mcool \
    regions_all.bed -r region4,region5 -b 64

Example loop merging:

Example fracshift:
