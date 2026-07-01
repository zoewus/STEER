import torch
import numpy as np
from .sample import ddpm_tsr_swapped

@torch.no_grad()
def mixture_log_likelihood(x, means=None, stds=None, weights=None):
    """Score each sample under N(0,1) — higher is better."""
    return torch.distributions.Normal(0, 1).log_prob(x).squeeze(-1)  # [N]

@torch.no_grad()
def best_of_n(model, dataset_config, lam, lam_ladder, n_samples=8, swap_algorithm=None, replica_swaps=False):
    means = dataset_config["means"]
    stds  = dataset_config["stds"]
    shape = dataset_config["dataset_shape"]

    best_x     = None
    best_score = -float("inf")
    all_scores = []

    for n in range(n_samples):
        torch.manual_seed(n)
        np.random.seed(n)

        x_ladder = ddpm_tsr_swapped(
            model=model,
            dataset_shape=shape,
            lam=lam,
            lam_ladder=lam_ladder,
            replica_swaps=replica_swaps,
            swap_algorithm=swap_algorithm,
        )

        # Use lam=1.0 replica as candidate
        x_candidate = x_ladder[lam]  # [N, 1]

        ll = mixture_log_likelihood(x_candidate, means, stds)
        mean_ll = ll.mean().item()
        all_scores.append(mean_ll)

        if mean_ll > best_score:
            best_score = mean_ll
            best_x = {lam_val: x_ladder[lam_val].clone() for lam_val in lam_ladder}

        print(f"  Run {n+1}/{n_samples}  mean_ll={mean_ll:.4f}  best_so_far={best_score:.4f}")

    print(f"\nBest run score: {best_score:.4f}  (across {n_samples} runs)")
    return best_x, best_score, all_scores