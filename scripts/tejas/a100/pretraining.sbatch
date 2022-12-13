#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-23:30:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=convirt_pretrain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tm3647@nyu.edu

singularity exec --nv --overlay $SCRATCH/public/mlh-env-50G-10M.ext3:ro --overlay /scratch/tm3647/public/CheXpert_v1.sqf:ro --overlay /scratch/tm3647/public/Mimic_cxr_jpg.sqf:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh;

conda activate /ext3/fedml-env/;

cd /scratch/tm3647/MLH/ConVIRT-private/;

python src/pretrain.py
"