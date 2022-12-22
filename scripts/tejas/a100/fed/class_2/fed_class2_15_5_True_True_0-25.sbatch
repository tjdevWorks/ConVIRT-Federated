#!/bin/bash
    
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --time=0:15:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=fed_class2_15_5_True_True_0-25
#SBATCH --output=/scratch/tm3647/MLH/ConVIRT-private/sbatch_outs/%j_0_log.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tm3647@nyu.edu

    singularity exec --nv --overlay /scratch/tm3647/public/mlh-env-50G-10M.ext3:ro --overlay /scratch/tm3647/public/CheXpert_v1.sqf:ro --overlay /scratch/tm3647/public/Mimic_cxr_jpg.sqf:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh;

conda activate /ext3/fedml-env/;

cd /scratch/tm3647/MLH/ConVIRT-private/;

python src/run_simulation.py --config-name=prod_simulation server_config.num_rounds=15 pool_size=5 partitions=class partitions.num_clients=5 partitions.exclusive=True partitions.equal_num_samples=True task_name='fed_chexpert_class2' job_name=fed_class2_15_5_True_True_0-25 datamodule.batch_size=256 strategy.fraction_fit=0.25 sample_percent=0.01
"