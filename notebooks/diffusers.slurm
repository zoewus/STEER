#!/bin/bash
#SBATCH --job-name=diffusers
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:20:00
#SBATCH --mem=16G
#SBATCH --output=output_files/diffusers.out
#SBATCH --error=output_files/diffusers.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate diffusers

srun python ~/STEER/diffusers/exchange_scripts/generate_samples.py \
    --lam-start 0.85 --lam-end 1.15 --n-particles 4 \
    --tsr-path "images/tsr" \
    --steer-path "images/steer" \
    --index_until 100