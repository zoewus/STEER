from diffusers import StableDiffusion3Pipeline
import torch
from pathlib import Path
import gc
import pandas as pd
from fid import compute_sweep
import argparse
import os 


parser = argparse.ArgumentParser()
parser.add_argument("--lam-start", type=float, default=1.0)
parser.add_argument("--lam-end", type=float, default=1.0)
parser.add_argument("--n-particles", type=int, default=4)
parser.add_argument("--tsr-path", type=Path, default="/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/tsr")
parser.add_argument("--steer-path", type=Path, default="/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/steer_sym__001_04")
parser.add_argument("--index_until", type=int, default=5)
args = parser.parse_args()

lam_start = args.lam_start
lam_end = args.lam_end
N_PARTICLES = args.n_particles
TSR_DIR    = args.tsr_path
STEER_DIR = args.steer_path
INDEX_UNTIL = args.index_until

LAM_VALUES = [0.85, 0.95, 1.05, 1.15]

REAL_DIR    = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/laion_5k_real"
PROMPTS_FILE = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/data_files/laion_5k_prompts.csv")
MODEL_CACHE = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/model_checkpoints"
SEED = 42

TSR_SIGMA = 1.0

N_INF_STEPS = 30

GUIDANCE_SCALE = 7.5

pipe = StableDiffusion3Pipeline.from_pretrained(
	"stabilityai/stable-diffusion-3-medium-diffusers",
	torch_dtype=torch.float16,
	cache_dir=MODEL_CACHE
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

gc.collect()
torch.cuda.empty_cache()

prompts = pd.read_csv(PROMPTS_FILE, usecols=["text"], nrows=INDEX_UNTIL)["text"].tolist()
print(f"Loaded {len(prompts)} prompts")

replica_exchanges = [True]

lam_dirs = {}
for re in replica_exchanges:
	base = STEER_DIR if re else TSR_DIR
	lam_dirs[re] = {l: base / f"lam{l:.3f}".replace(".", "p") for l in LAM_VALUES}
	for d in lam_dirs[re].values():
		d.mkdir(parents=True, exist_ok=True)

for idx, prompt in enumerate(prompts):

	for replica_exchange in replica_exchanges:

		generator = torch.Generator(device="cuda").manual_seed(SEED)

		if replica_exchange == True:
			particles = N_PARTICLES
		else:
			particles = 1

		images = pipe(
			prompt,
			negative_prompt="",
			num_inference_steps=N_INF_STEPS,
			guidance_scale=GUIDANCE_SCALE,
			lam_start=lam_start,
			lam_end=lam_end,
			replica_exchange=replica_exchange,
			n_particles=particles,
			generator=generator,
		).images

		images[0].save(lam_dirs[replica_exchange][LAM_VALUES[0]] / f"{idx:05d}.png", icc_profile=None)
		images[1].save(lam_dirs[replica_exchange][LAM_VALUES[1]] / f"{idx:05d}.png", icc_profile=None)
		images[2].save(lam_dirs[replica_exchange][LAM_VALUES[2]] / f"{idx:05d}.png", icc_profile=None)
		images[3].save(lam_dirs[replica_exchange][LAM_VALUES[3]] / f"{idx:05d}.png", icc_profile=None)

		del images
		torch.cuda.empty_cache()

print("All lam values complete.")

os.environ["HF_TOKEN"] = "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"
tsr_results = compute_sweep(
    replica_exchanges=[True, False],
    device="cuda",
    target_indices=None,
    index_until=INDEX_UNTIL,
    tsr_dir=TSR_DIR,
    pt_tsr_dir=STEER_DIR,        # fix kwarg name
    lam_values_ptsr=LAM_VALUES,
    lam_values_tsr=LAM_VALUES,
)