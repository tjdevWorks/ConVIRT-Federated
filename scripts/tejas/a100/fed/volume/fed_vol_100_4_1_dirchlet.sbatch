#!/bin/bash
    
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --time=4:30:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=fed_vol_100_4_1_dirchlet
#SBATCH --output=/scratch/tm3647/MLH/ConVIRT-private/sbatch_outs/%j_0_log.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tm3647@nyu.edu

    singularity exec --nv --overlay /scratch/tm3647/public/mlh-env-50G-10M.ext3:ro --overlay /scratch/tm3647/public/CheXpert_v1.sqf:ro --overlay /scratch/tm3647/public/Mimic_cxr_jpg.sqf:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh;

conda activate /ext3/fedml-env/;

cd /scratch/tm3647/MLH/ConVIRT-private/;

python src/run_simulation.py --config-name=prod_simulation server_config.num_rounds=100 pool_size=4 partitions.num_clients=4 partitions.scale=1 partitions.mode='dirchlet' task_name='fed_chexpert_volume' job_name=fed_vol_100_4_1_dirchlet datamodule.batch_size=256
"