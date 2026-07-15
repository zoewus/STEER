#!/bin/bash
#SBATCH --job-name=diffpolicy_sample
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:30:00
#SBATCH --mem=8G
#SBATCH --output=output_files/diffpolicy_sample_%j.out
#SBATCH --error=output_files/diffpolicy_sample_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate robomimic_venv

mkdir -p output_files
mkdir -p rollout_videos

# make sure offscreen rendering works headlessly
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

srun python robomimic/robomimic/scripts/run_trained_agent.py \
    --agent /n/netscratch/kempner_undergrads/Everyone/zwu/diffusion_policy_trained_models/square_ld_diffusion_no_replica_exchange/20260715081703/models/model_epoch_1000.pth \
    --n_rollouts 50 \
    --horizon 400 \
    --seed 0 \
    --video_path rollout_videos/square_ld_diffusion_no_replica_exchange_epoch1000.mp4 \
    --camera_names agentview robot0_eye_in_hand