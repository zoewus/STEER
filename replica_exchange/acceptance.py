import torch
import random

@torch.no_grad()
def _score_constant(a_bar, tsr_lam, eps=1e-6):
    return 1 / (a_bar * tsr_lam + (1 - a_bar))

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device, dtype):
    n_half = n_replicas // 2

    # s in (0, 1], excluding 0 -> avoids landing exactly on lam=1.0 twice
    s = torch.linspace(0, 1, n_half + 1, device=device, dtype=dtype)[1:]

    # geometric interpolation: lam = 1.0 * ratio^s, crowds points near lam=1.0
    lam_upper = lam_end ** s      # 1.0 -> lam_end
    lam_lower = lam_start ** s    # 1.0 -> lam_start

    if n_replicas % 2 == 0:
        lam_ladder = torch.cat([lam_lower.flip(0), lam_upper])
    else:
        # odd count: include exact midpoint lam=1.0 once
        one = torch.ones(1, device=device, dtype=dtype)
        lam_ladder = torch.cat([lam_lower.flip(0), one, lam_upper])

    return lam_ladder

def scale(grad, a_bar, lam_start, lam_end, n_replicas):
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)
    lam_ladder_t = _score_constant(a_bar, lam_ladder)                  # (n_replicas,) vectorized
    
    lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
    return grad * lam_ladder_t

def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas, eps_ladder, i=None, flow=None):
    x_out = x_ladder.clone()
    eps_out = eps_ladder.clone()

    step_val = (i - 1) if i is not None else int(t_val)

    if step_val % 2 == 0:
        attempt_num = step_val // 2
        n_pairs = n_replicas - 1
        cycle_num = attempt_num // n_pairs
        pos_in_cycle = attempt_num % n_pairs

        if cycle_num % 2 == 0:
            i_hot = (n_replicas - 2) - pos_in_cycle   # hot -> cold
        else:
            i_hot = pos_in_cycle                       # cold -> hot

        pairs = [(i_hot, i_hot + 1)]
    else:
        pairs = []

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

        x_diff = (x_tau - x_s)
        tsr_diff = 1/lam_ladder[index_t] -  1/lam_ladder[index_s] # this is always positive
        integral = (score_tau + score_s) * x_diff #* tsr_diff
        log_ratio = torch.clamp(integral.mean() , max=0.0)
        acceptance = torch.exp(log_ratio)        
        accept = (torch.rand_like(acceptance) < acceptance).float() 

        print(f"Time {t_val} {lam_ladder[index_t]:.2f} and {lam_ladder[index_s]:.2f} integral {integral.mean().item():.2f} acceptance {acceptance.mean().item()}  accept {accept.mean().item()}")

        x_out[sl] = accept * x_s + (1 - accept) * x_tau
        x_out[ss] = accept * x_tau + (1 - accept) * x_s
        eps_out[sl] = accept * eps_s + (1 - accept) * eps_tau
        eps_out[ss] = accept * eps_tau + (1 - accept) * eps_s

    return x_out, eps_out