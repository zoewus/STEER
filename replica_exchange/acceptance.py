import torch

@torch.no_grad()
def _score_constant(a_bar, tsr_lam):
    output = 1 / (a_bar * tsr_lam + 1 - a_bar)
    return output

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device=None, dtype=None):
    lam_ladder = torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)
    return lam_ladder

def scale(grad, a_bar, lam_start, lam_end, n_replicas):
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)
    lam_ladder_t = _score_constant(a_bar, lam_ladder)                  # (n_replicas,) vectorized
    
    lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
    return grad * lam_ladder_t

def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas, eps_ladder, i=None, flow=None):
    x_out = x_ladder.clone()
    eps_out = eps_ladder.clone()

    if i is not None:
        offset = i % 2
    else:
        offset = t_val % 2
    pairs = [(i_tau, i_tau + 1) for i_tau in range(offset, n_replicas - 1, 2)]
    index = x_ladder.shape[0] // n_replicas

    # interps = []
    # for i_pair, (index_t, index_s) in enumerate(pairs):
    #     x_tau = x_ladder[index * index_t : index * (index_t + 1)]
    #     x_s   = x_ladder[index * index_s : index * (index_s + 1)]
    #     interps.append((x_tau + x_s) / 2)
    
    # x_interps = torch.cat(interps, dim=0)
    #eps_interps = compute_eps(x_interps, t_val)

    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=x_ladder.device, dtype=x_ladder.dtype)

    if flow:
        # convert v-prediction -> eps for the acceptance math (SD3 / flow-matching models)
        sigma_t = 1 - a_bar
        x0_hat_ladder = x_ladder - sigma_t * eps_ladder
        score_ladder = (x_ladder - (1 - sigma_t) * x0_hat_ladder) / sigma_t
    else:
        # non-flow models predict eps directly
        score_ladder = - eps_ladder / (1 - a_bar) ** 0.5
        # score_interps = - eps_interps / (1 - a_bar) ** 0.5

    for i_pair, (index_t, index_s) in enumerate(pairs):
        x_tau = x_ladder[index * index_t : index * (index_t + 1)]
        x_s   = x_ladder[index * index_s : index * (index_s + 1)]

        score_tau = score_ladder[index * index_t : index * (index_t + 1)]
        score_s   = score_ladder[index * index_s : index * (index_s + 1)]
        # score_interp = score_interps[index * i_pair : index * (i_pair + 1)]

        eps_tau = eps_ladder[index * index_t : index * (index_t + 1)]
        eps_s   = eps_ladder[index * index_s : index * (index_s + 1)]

        tsr_diff = _score_constant(a_bar, lam_ladder[index_t]) - _score_constant(a_bar, lam_ladder[index_s])
        integral = (score_s   + score_tau) * (x_tau - x_s) * tsr_diff / 6

        log_acceptance = torch.clamp(integral.sum(), max=0)
        u = torch.rand_like(log_acceptance)
        accept = (torch.log(u) < log_acceptance).float()

        x_out[index * index_t : index * (index_t + 1)] = accept * x_s   + (1 - accept) * x_tau
        x_out[index * index_s : index * (index_s + 1)] = accept * x_tau + (1 - accept) * x_s

        eps_out[index * index_t : index * (index_t + 1)] = accept * eps_s   + (1 - accept) * eps_tau
        eps_out[index * index_s : index * (index_s + 1)] = accept * eps_tau + (1 - accept) * eps_s

        print(f"t {t_val} | lam {lam_ladder[index_t]:.2f}↔{lam_ladder[index_s]:.2f} "
                f"| integral {integral.mean():.3f} | acceptance {accept.mean():.3f}")

    return x_out, eps_out