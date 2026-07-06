#!/bin/bash
#SBATCH --job-name=maze
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH --output=output_files/maze.out
#SBATCH --error=output_files/maze.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate /n/netscratch/kempner_undergrads/Everyone/zwu/environments/maze

srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.95 --lam_end 1.05 --n_particles 4 --device cuda
