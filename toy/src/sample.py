import torch
import numpy as np

from .schedule import betas, alphas, alpha_bars, ts_desc
from .config_toy import DEVICE, CKPT_DIR

from .acceptance import swap, scale, init_temp_idx

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR


@torch.no_grad()
def ddpm_tsr_swapped(model, dataset_shape, lam_start, lam_end, n_replicas, replica_swaps=False, swap_algorithm=None):

	x_ladder = []
	x_init = torch.randn(dataset_shape, device=device).sort().values
	for i in range(n_replicas):
		x_ladder.append(x_init)  # interpolate between x_init and independent noise
	x_ladder = torch.cat(x_ladder, dim=0)

	temp_idx = init_temp_idx(n_replicas, *x_ladder.shape[1:], device)

	for t in ts_desc:

		alpha_t = alphas[t]
		beta_t = betas[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)
		a_bar = alpha_bars[t]

		for i in range(n_replicas):
			noise = torch.randn(dataset_shape, device=device)

			x = x_ladder[i * dataset_shape[0] : (i+1) * dataset_shape[0]].clone()
			ones = torch.ones((x.shape[0], 1), device=x.device)

			eps_hat_original = model(x, t * ones)

			eps_hat = scale(x, eps_hat_original, a_bar, lam_start, lam_end, n_replicas, temp_idx=None)

			score_hat = -eps_hat / torch.sqrt(1 - a_bar)
			x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

			x_ladder[i * dataset_shape[0] : (i+1) * dataset_shape[0]] = x.clone()


		if replica_swaps :
			
			temp_idx = swap(x_ladder, t, a_bar, 0.98, 1.02, 4, eps_hat_original, temp_idx)

	return x_ladder