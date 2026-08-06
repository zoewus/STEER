#!/bin/bash
#SBATCH --job-name=diffuser_s
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=0-02:00:00
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=output_files/diffusers_09_%a.out
#SBATCH --error=output_files/diffusers_09_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate diffusers

# One lambda value per array task -> one GPU per lambda, all running in parallel.
LAM_VALUES=(1.05 1.1)
LAM=${LAM_VALUES[$SLURM_ARRAY_TASK_ID]}

srun python ~/STEER/diffusers/exchange_scripts/generate_samples.py \
    --lam "$LAM" \
    --index-until 500\
    --steer-path "images/steer_x0_scaling_and_eps"