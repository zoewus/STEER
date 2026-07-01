#!/bin/bash
#SBATCH --job-name=training_composed
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-5:00:00
#SBATCH --mem=16G
#SBATCH --output=output_files/training_composed.out
#SBATCH --error=output_files/training_composed.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate lightweight_tester

cd 1D
srun python -m src.train