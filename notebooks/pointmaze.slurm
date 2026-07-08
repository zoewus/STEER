#!/bin/bash
#SBATCH --job-name=maze
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=0-02:00:00
#SBATCH --mem=16G
#SBATCH --output=output_files/maze.out
#SBATCH --error=output_files/maze.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
conda activate maze

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 1.00 --lam_end 1.00 --n_particles 4 --device cuda

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.95 --lam_end 1.050 --n_particles 4 --device cuda
MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.95 --lam_end 1.05 --replica_exchange --n_particles 4 --device cuda

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.85 --lam_end 1.15 --n_particles 4 --device cuda
MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.85 --lam_end 1.15 --replica_exchange --n_particles 4 --device cuda

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.75 --lam_end 1.25 --n_particles 4 --device cuda
MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.75 --lam_end 1.25 --replica_exchange --n_particles 4 --device cuda

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.65 --lam_end 1.35 --n_particles 4 --device cuda
MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.65 --lam_end 1.35 --replica_exchange --n_particles 4 --device cuda


MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 1.00 --lam_end 1.00 --n_particles 8 --device cuda

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.95 --lam_end 1.05 --n_particles 8 --device cuda
MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.95 --lam_end 1.05 --replica_exchange --n_particles 8 --device cuda

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.85 --lam_end 1.15 --n_particles 8 --device cuda
MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.85 --lam_end 1.15 --replica_exchange --n_particles 8 --device cuda

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.75 --lam_end 1.25 --n_particles 8 --device cuda
MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.75 --lam_end 1.25 --replica_exchange --n_particles 8 --device cuda

MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.65 --lam_end 1.35 --n_particles 8 --device cuda
MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py --dataset pointmaze-giant-navigate-v0 --lam_start 0.65 --lam_end 1.35 --replica_exchange --n_particles 8 --device cuda


