#!/bin/sh

#SBATCH -n 16 -o output/cache_womd_validation.log-%j

export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

# DATA_SPLIT=validation # training, validation, testing
# DATA_SPLIT=training # training, validation, testing
DATA_SPLIT=testing # training, validation, testing


#source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

source activate catk

export PATH=/home/gridsan/thlin/.conda/envs/catk/bin:$PATH   # use torchrun in conda bin

python \
  -m src.data_preprocess \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --input_dir ../data/waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario \
  --output_dir ../data/waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/cache/SMART