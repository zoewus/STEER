"""
Generate images for a TSR baseline (no particle exchange) and a replica-exchange
"steer" sweep across multiple n_particles / lambda-schedule settings.

TSR:   one independent single-particle generation per lambda value. n_particles
       is irrelevant here, so each image is produced by its own pipe call with
       replica_exchange=False, n_particles=1.

Steer: for each requested particle count, build a lambda schedule via
       np.linspace(lam_start, lam_end, n_particles), run one replica-exchange
       pipe call per prompt, and save every particle's image. Output lands in
       steer_path/steer_{n_particles}/lam.../
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusion3Pipeline

# ----------------------------------------------------------------------------
# Fixed experiment config
# ----------------------------------------------------------------------------
REAL_DIR = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/laion_5k_real")
PROMPTS_FILE = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/data_files/laion_5k_prompts.csv")
MODEL_CACHE = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/model_checkpoints"
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

SEED = 42
N_INF_STEPS = 30
GUIDANCE_SCALE = 7.5
LAM_END = 1.0


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Starting lambda value. lam_end is always fixed at 1.0.",
    )
    parser.add_argument(
        "--n-particles-list",
        type=int,
        nargs="+",
        default=[4],
        help="One or more particle counts to sweep for the steer (replica-exchange) runs.",
    )
    parser.add_argument(
        "--tsr-path",
        type=Path,
        default=Path("images/tsr"),
    )
    parser.add_argument(
        "--steer-path",
        type=Path,
        default=Path("images/steer"),
    )
    parser.add_argument("--index-until", type=int, default=5)
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def lam_dirname(lam: float) -> str:
    return f"lam{lam:.3f}".replace(".", "p")


def load_pipe() -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_prompts(index_until: int) -> list[str]:
    prompts = pd.read_csv(PROMPTS_FILE, usecols=["text"], nrows=index_until)["text"].tolist()
    print(f"Loaded {len(prompts)} prompts")
    return prompts


def free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# TSR: n_particles-free, one lambda at a time
# ----------------------------------------------------------------------------
def run_tsr(pipe, prompts: list[str], lam: float, tsr_dir: Path) -> None:
    lam_dir = tsr_dir / lam_dirname(lam)
    lam_dir.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        out_path = lam_dir / f"{idx:05d}.png"
        if out_path.exists():
            print(f"[tsr] skipping idx={idx} (already exists at {out_path})")
            continue

        generator = torch.Generator(device="cuda").manual_seed(SEED)

        image = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=N_INF_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            lam_start=lam,
            lam_end=lam,
            replica_exchange=False,
            n_particles=1,
            generator=generator,
        ).images[0]

        image.save(out_path, icc_profile=None)
        del image

        free_cuda()

    print("TSR generation complete.")


# ----------------------------------------------------------------------------
# Steer: replica exchange, swept over n_particles
# ----------------------------------------------------------------------------
def run_steer(
    pipe,
    prompts: list[str],
    n_particles_list: list[int],
    lam_start: float,
    lam_end: float,
    steer_dir: Path,
) -> None:
    for n_particles in n_particles_list:
        run_dir = steer_dir / f"steer_{n_particles}" / lam_dirname(lam_start)
        run_dir.mkdir(parents=True, exist_ok=True)

        for idx, prompt in enumerate(prompts):
            out_path = run_dir / f"{idx:05d}.png"
            if out_path.exists():
                print(f"[steer_{n_particles}] skipping idx={idx} (already exists at {out_path})")
                continue

            generator = torch.Generator(device="cuda").manual_seed(SEED)

            images = pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=N_INF_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                lam_start=lam_start,
                lam_end=lam_end,
                replica_exchange=True,
                n_particles=n_particles,
                generator=generator,
            ).images

            images[0].save(out_path, icc_profile=None)

            del images
            free_cuda()

        print(f"Steer generation complete for n_particles={n_particles}.")
 
# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    pipe = load_pipe()
    free_cuda()

    prompts = load_prompts(args.index_until)

    # TSR uses the finest lambda grid across all requested particle counts,
    # since it doesn't depend on n_particles itself.

    if args.lam <1:
        lam_end =args.lam+0.2
    else:
        lam_end = args.lam+0.1

    run_tsr(pipe, prompts, args.lam, args.tsr_path)
    run_steer(pipe, prompts, args.n_particles_list, args.lam, lam_end, args.steer_path)

    print("All runs complete.")


if __name__ == "__main__":
    main()