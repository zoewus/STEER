import torch

@torch.no_grad()
def _score_constant(a_bar, tsr_lam, eps=1e-6):
    return 1 / (a_bar * tsr_lam + (1 - a_bar))

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device, dtype, p=3.0):
    t = torch.linspace(0, 1, n_replicas, device=device, dtype=dtype)
    t = t ** p  # larger p -> more points crowd near lam_start
    # lam_ladder = torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)
    lam_ladder = lam_start + (lam_end - lam_start) * t
    return lam_ladder

def scale(grad, a_bar, lam_start, lam_end, n_replicas):
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)
    lam_ladder_t = _score_constant(a_bar, lam_ladder)                  # (n_replicas,) vectorized
    
    lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
    return grad * lam_ladder_t

def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas, eps_ladder, i=None, flow=None):
    x_out = x_ladder.clone()
    eps_out = eps_ladder.clone()

    offset = int(i % 2) if i is not None else int(t_val) % 2
    pairs = [(i_tau, i_tau + 1) for i_tau in range(offset, n_replicas - 1, 2)]
    index = x_ladder.shape[0] // n_replicas

    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=x_ladder.device, dtype=x_ladder.dtype)

    if flow:
        sigma_t = 1 - a_bar
        x0_hat_ladder = x_ladder - sigma_t * eps_ladder
        score_ladder = (x_ladder - (1 - sigma_t) * x0_hat_ladder) / sigma_t
    else:
        score_ladder = - eps_ladder / (1 - a_bar) ** 0.5

    for index_t, index_s in pairs:
        sl = slice(index * index_t, index * (index_t + 1))
        ss = slice(index * index_s, index * (index_s + 1))

        x_tau, x_s = x_ladder[sl], x_ladder[ss]
        score_tau, score_s = score_ladder[sl], score_ladder[ss]
        eps_tau, eps_s = eps_ladder[sl], eps_ladder[ss]

        tsr_tau = _score_constant(a_bar, lam_ladder[index_t])
        integral = 0.5* (score_tau + score_s) * (x_s - x_tau)
        acceptance = torch.exp(integral.sum())
        accept = (torch.rand_like(acceptance) < acceptance).float() 

        print(f"Time {t_val} {lam_ladder[index_t]:.2f} and {lam_ladder[index_s]:.2f} integral {integral.mean().item():.2f} acceptance {accept.mean().item()}")

        x_out[sl] = accept * x_s + (1 - accept) * x_tau
        x_out[ss] = accept * x_tau + (1 - accept) * x_s
        eps_out[sl] = accept * eps_s + (1 - accept) * eps_tau
        eps_out[ss] = accept * eps_tau + (1 - accept) * eps_s

    return x_out, eps_out