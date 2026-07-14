#!/bin/bash
#SBATCH --job-name=maze
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=0-06:00:00
#SBATCH --mem=10G
#SBATCH --output=output_files/maze_%A_%a.out
#SBATCH --error=output_files/maze_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
conda activate maze

mkdir -p output_files

N_REPEATS=10

# Each entry: lam_start lam_end n_particles
CONFIGS=(
    "0.95 1.05 4"
    "0.95 1.1  4"
    "0.95 1.15 4"
    "0.9  1.05 4"
    "0.95 1.05 8"
    "0.95 1.1  8"
    "0.95 1.15 8"
    "0.9  1.05 8"
)

for cfg in "${CONFIGS[@]}"; do
    read -r lam_start lam_end n_particles <<< "$cfg"
    tag="ls${lam_start}_le${lam_end}_np${n_particles}"

    for mode in noRE RE; do
        if [ "$mode" == "RE" ]; then
            re_flag="--replica_exchange"
        else
            re_flag=""
        fi

        for rep in $(seq 1 "$N_REPEATS"); do
            echo "=== Running ${tag} mode=${mode} rep=${rep} ==="
            MUJOCO_GL=egl srun python ~/STEER/pointmaze/run.py \
                --dataset pointmaze-giant-navigate-v0 \
                --lam_start "$lam_start" --lam_end "$lam_end" \
                --n_particles "$n_particles" --device cuda \
                $re_flag \
                > "output_files/${tag}_${mode}_rep${rep}.log" 2>&1
        done
    done
done