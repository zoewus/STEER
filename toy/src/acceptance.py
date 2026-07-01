import torch

@torch.no_grad()
def _score_constant(a_bar, tsr_lam):
	output = 1/(a_bar* tsr_lam + (1-a_bar))
	return output

@torch.no_grad()
def swap(x_ladder, lam_ladder, t_val, a_bar, compute_eps, chunk_size=50000):
	offset = t_val % 2
	n_replicas = len(lam_ladder)
	pairs = [(i, i+1) for i in range(offset, n_replicas - 1, 2)]
	index = x_ladder.shape[0] // n_replicas

	# collect all sorted pairs
	all_x_tau, all_x_s = [], []
	sorted_idxs_tau, sorted_idxs_s = [], []
		
	for index_t, index_s in pairs:
			x_tau = x_ladder[index*index_t : index*(index_t+1)].clone()
			x_s   = x_ladder[index*index_s : index*(index_s+1)].clone()
			bs = x_tau.shape[0]

			idx_tau = torch.argsort(x_tau.reshape(bs, -1).squeeze(-1), stable=True)
			idx_s   = torch.argsort(x_s.reshape(bs, -1).squeeze(-1),   stable=True)

			all_x_tau.append(x_tau[idx_tau])
			all_x_s.append(x_s[idx_s])
			sorted_idxs_tau.append(idx_tau)
			sorted_idxs_s.append(idx_s)

	# single forward pass over all pairs
	x_tau_cat = torch.cat(all_x_tau, dim=0)
	x_s_cat   = torch.cat(all_x_s,   dim=0)
	x_both    = torch.cat([x_tau_cat, x_s_cat], dim=0)

	chunks = x_both.split(chunk_size, dim=0)
	scores = torch.cat([compute_eps(c, t_val) for c in chunks], dim=0)
	score_tau_all, score_s_all = scores.chunk(2, dim=0)

	# apply swaps
	x_ladder = x_ladder.clone()
	for i, (index_t, index_s) in enumerate(pairs):
		sl = slice(i * index, (i+1) * index)
		score_tau = score_tau_all[sl] 
		score_s   = score_s_all[sl] 
		x_tau_s   = all_x_tau[i]
		x_s_s     = all_x_s[i]

		tsr_diff = _score_constant(a_bar, lam_ladder[index_t]) - _score_constant(a_bar, lam_ladder[index_s])
		integral = 0.5 * (score_s + score_tau) * (x_tau_s - x_s_s) * tsr_diff

		acceptance = (torch.rand_like(integral) < torch.exp(integral)).float()
		x_tau_new = acceptance * x_s_s   + (1 - acceptance) * x_tau_s
		x_s_new   = acceptance * x_tau_s + (1 - acceptance) * x_s_s

		x_ladder[index*index_t : index*(index_t+1)] = x_tau_new[torch.argsort(sorted_idxs_tau[i], stable=True)]
		x_ladder[index*index_s : index*(index_s+1)] = x_s_new[torch.argsort(sorted_idxs_s[i],   stable=True)]

		print(f"t {t_val} | lam {lam_ladder[index_t]:.2f}↔{lam_ladder[index_s]:.2f} | integral {integral.mean():.3f} | acceptance {acceptance.mean():.3f}")

	return x_ladder
