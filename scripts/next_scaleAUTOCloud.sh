#!/bin/bash
#SBATCH -n 8 --gres=gpu:volta:1 -o output/next_scale_autoreg-00015_VAR_allGrad_ang_acc100x_bfloat16_NofinetuneVqvae.log-%j


export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

# source /etc/profile
# module unload anaconda
# PYTHON=/usr/bin/python3.11
# export PYTHONPATH=/home/gridsan/thlin/.conda/envs/catk/lib/python3.11/site-packages:$PYTHONPATH

# source activate catk

module load anaconda/Python-ML-2024b
source activate var_catk

export PATH=/home/gridsan/thlin/.conda/envs/catk/bin:$PATH   # use torchrun in conda bin
# alias torchrun='/home/gridsan/thlin/.conda/envs/catk/bin/torchrun'

# /home/gridsan/thlin/.conda/envs/catk/bin/torchrun \

ulimit -n # check current open files limit
ulimit -n 65536
ulimit -n
export COLUMNS=200 # increase hydra text length

MY_EXPERIMENT="next_scale_autoreg"

export MASTER_PORT=44144
export MASTER_ADDR=127.0.0.1

while true
do
    status="$(nc -z 127.0.0.1 $MASTER_PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
    echo "MASTER_PORT already in use, retrying...$MASTER_PORT"
    MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
done
echo $MASTER_PORT

################################
# MY_EXPERIMENT="pre_bc"
# MY_EXPERIMENT="clsft"
MY_EXPERIMENT="next_scale_autoreg"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_n_step16"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_loss_L2_traj"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_loss_L2_traj_gradFix"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_loss_L2_traj_gradFix_noXEntropyLoss"
MY_TASK_NAME=$MY_EXPERIMENT"-vqvae_loss_vqvae_logits_gradFix"
MY_TASK_NAME=$MY_EXPERIMENT"-0003_vqvae_loss_using_logits__loss_sum_fix"
MY_TASK_NAME=$MY_EXPERIMENT"-0003-2_vocabSize4096_Cvae64_quantizeClass"
MY_TASK_NAME=$MY_EXPERIMENT"-0004_vocabSize4096_Cvae64_vqvae_varClass"
MY_TASK_NAME=$MY_EXPERIMENT"-0005_VAR_32b"
MY_TASK_NAME=$MY_EXPERIMENT"-0007_VAR_auto_32b_overflow"
MY_TASK_NAME=$MY_EXPERIMENT"-0007-2_VAR_auto_32b_overflow_schedulerLR"
MY_TASK_NAME=$MY_EXPERIMENT"-0007-2b_VAR_auto_32b_overflow_schedulerLR"
# MY_TASK_NAME=$MY_EXPERIMENT"-0007-3_VAR_auto_32b_sample12_VARout"
# MY_TASK_NAME=$MY_EXPERIMENT"-0008_VAR_bfloat16_finetune_vqvae"
# MY_TASK_NAME=$MY_EXPERIMENT"-0008-2_VAR_bfloat16_finetune_vqvae_schedulerLR"
MY_TASK_NAME=$MY_EXPERIMENT"-0009_VAR_bfloat16_finetune_vqvae_LR1e-5"
MY_TASK_NAME=$MY_EXPERIMENT"-0009b_VAR_bfloat16_finetune_vqvae_LR1e-5"
MY_TASK_NAME=$MY_EXPERIMENT"-00010_VAR_ang_acc_bfloat16_finetune_vqvae"
MY_TASK_NAME=$MY_EXPERIMENT"-00011b_VAR_ang_acc100x_bfloat16_finetune_vqvae"
MY_TASK_NAME=$MY_EXPERIMENT"-00012_VAR_ang_acc10x_bfloat16_finetune_vqvae"
MY_TASK_NAME=$MY_EXPERIMENT"-00013_VAR_ang_acc100x_bfloat16_NofinetuneVqvae"
# MY_TASK_NAME=$MY_EXPERIMENT"-00014_VAR_ang_acc100x_32bit_NofinetuneVqvae"
MY_TASK_NAME=$MY_EXPERIMENT"-00015_VAR_allGrad_ang_acc100x_bfloat16_NofinetuneVqvae"
# MY_TASK_NAME=$MY_EXPERIMENT"-testing"

torchrun \
  -m \
  --master-port $MASTER_PORT \
  src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  +config_name=run_scale_autoreg.yaml  # override @hydra.main(config_path="../configs/", config_name="run.yaml"