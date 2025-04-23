Loop calling and centering with CHIRON+fracshift


## Overview

This repo contains the code for CHIRON, a CNN that was built to classify loops in high-depth RCMC, and RCMC fracshift, which centers loops using sub-pixel localization methods. 

## Installation and usage:

The required packages for CHIRON and fracshift can be installed from the attached environment.yml. CHIRON runs on tensorflow and can be CUDA-configured for GPU usage. However, the model can be run on CPUs especially if you are imputing smaller (<3Mb) regions.

The complete usage for CHIRON + fracshift is to impute loops on your RCMC, merge them via local max finding, and then center them with fracshift. You could also use any of these steps separately (ex. just apply fracshift on existing loop calls), but the input formats track between all three so check the documentation for how to prepare your input for each step separately.

Example usage of CHIRON -> merge -> fracshift:

#### call loops
python /chiron/chiron/imputation.py CHIRON_v0 \
    GM12878 GM12878_merged_realigned.50.mcool \
    region_list.txt -r region4,region5 -b 64 \
    -o test_output

#### merge loops 
python /chiron/chiron/merge_loops_gaussian.py chiron_out/CHIRON_v0/GM12878_region4,region5_loops_raw.txt region_list.txt

#### fracshift (see also fracshift_exec_demo.py for more detail)
python /chiron/fracshift/exec_centering.py region_list.txt chiron_out/input_calls.txt chiron_out/corrected_calls.txt

## Loop calling with CHIRON:

Find below detailed instructions on training and running the model (note that most users can jump right to imputation and loop merging):

1. Data fetching and pre-processing scripts (preprocessing/) \n
get_pretraining_data.py \n
get_neg_pretraining_data.py \n

get_training_data.py \n
get_neg_training_data.py \n 

The negative data scripts rely on the corresponding positive data scripts, but the pretraining and fine-tuning data can be generated independently.

2. Training and imputation: (chiron/) 

a. Pre-training: model_pretrain.py \n
b. Fine-tuning: model_finetune.py \n
c. Imputation: imputation_by_region.py \n
d. Loop merging: merge_loops_gaussian.py \n

Until the fine-tuning step, all the scripts contain hard-coded paths and should be modified for each user.The fine-tuning, imputation, and loop merging can be run by specifying parameters in the command line flags (see -h flag for help).

3. fracshift (fracshift/):

After calling loops, there is an optional step to localize the loop centers to a theoretical "base pair" resolution (in practice, it will be limited by your read depth to about 0.1x the bin size at which the matrix is "filled in" i.e. there is sufficient signal in most of the bins). The fracshift application runs a user-facing script that runs through all the loops and applies fracshift; this script runs a few custom modules that implement all the functions.

User-facing scripts:
a. fracshift_exec_demo.py: shell that runs a demo of exec_centering.py  \n
b. exec_centering.py: main script that reads all the loops and applies fracshift.  \n

Modules:
c. centering.py: contains all centering/upresolution functions  \n
d. setup.py  \n
e. utils.py  \n

Required inputs:
- 'region_list': a 2-column file with names for the regions and the UCSC genome coordinates for each
- 'loop_path': path to the loopcalls which should be tab-delimited point calls, as in the output of CHIRON.

## More Usage:

Example pre-training (trains a 3-layer CNN):
python /chiron/chiron/model_pretrain.py 3 "MODEL_SAVE_NAME" "PRETRAIN_DATA_DIR" -l 0.001 -e 100 -b 64

Example fine-tuning (finetunes the last hidden layer+ of the previous pretrained model):
python /chiron/chiron/model_finetune.py "MODEL_SAVE_NAME" "FINETUNE_TAG" UNFREEZE -u -6 -d 0.3 -b 512 -e 50 -l 0.001 -g True

Example imputation:
python /chiron/chiron/imputation.py CHIRON_v0 \
    GM12878 GM12878_merged_realigned.50.mcool \
    regions_all.bed -r region4,region5 -b 64
