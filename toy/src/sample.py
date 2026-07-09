import torch
import numpy as np

from .schedule import betas, alphas, alpha_bars, ts_desc
from .config_toy import DEVICE, CKPT_DIR

from .acceptance import swap, _score_constant

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR


@torch.no_grad()
def ddpm_tsr_swapped(model, dataset_shape, lam_ladder, n_replicas=2, replica_swaps=False, swap_algorithm=None):

	x_ladder = []
	x_init = torch.randn(dataset_shape, device=device).sort().values
	for i, lam in enumerate(lam_ladder):
		x_ladder.append(x_init*np.sqrt(lam))  # interpolate between x_init and independent noise
	x_ladder = torch.cat(x_ladder, dim=0)

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

			eps_hat = model(x, t * ones)

			tsr_lam = lam_ladder[i]

			tsr = _score_constant(a_bar, tsr_lam)

			score_hat = -eps_hat * tsr / torch.sqrt(1 - a_bar)
			x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

			x_ladder[i * dataset_shape[0] : (i+1) * dataset_shape[0]] = x.clone()


		if replica_swaps :


			def compute_eps(x_t, t_int):
				orig_shape = x_t.shape
				bs = orig_shape[0]
				dtype = next(model.parameters()).dtype
				ones = torch.ones((bs, 1), device=x_t.device)
				t_in = t_int * ones
				eps_hat = model(x_t.to(dtype), t_in)
				return -eps_hat / torch.sqrt(1 - a_bar)
			
			x_ladder = swap(x_ladder, lam_ladder, t, a_bar,  compute_eps)

	return x_ladder