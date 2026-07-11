import torch
import random

@torch.no_grad()
def _score_constant(a_bar, tsr_lam, eps=1e-6):
    return 1 / (a_bar * tsr_lam + (1 - a_bar))

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device, dtype, bias=3.0):
    n_lower = n_replicas // 2
    n_upper = n_replicas - n_lower  # absorbs the odd remainder, if any

    # s in (0, 1], excluding 0 -> lam never lands exactly on 1.0
    s_lower = torch.linspace(0, 1, n_lower + 1, device=device, dtype=dtype)[1:]
    s_upper = torch.linspace(0, 1, n_upper + 1, device=device, dtype=dtype)[1:]

    # bias s toward 0 (bias > 1 compresses points near s=0, i.e. near lam=1;
    # s=1 is a fixed point so the far endpoints lam_start/lam_end are unchanged)
    s_lower = s_lower ** bias
    s_upper = s_upper ** bias

    lam_lower = lam_start ** s_lower   # approaches 1.0 -> lam_start
    lam_upper = lam_end ** s_upper     # approaches 1.0 -> lam_end

    lam_ladder = torch.cat([lam_lower.flip(0), lam_upper])

    return lam_ladder

def scale(grad, a_bar, lam_start, lam_end, n_replicas):
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)
    lam_ladder_t = _score_constant(a_bar, lam_ladder)                  # (n_replicas,) vectorized
    
    lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
    return grad * lam_ladder_t

def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas, eps_ladder, i=None, flow=None):
    x_out = x_ladder.clone()
    step_val = i
    if i % 2 ==0:
        offset = (step_val % 4)//2
        pairs = [(i_tau, i_tau + 1) for i_tau in range(offset, n_replicas - 1, 2)]
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

        tsr_diff = _score_constant(a_bar,lam_ladder[index_t]) - _score_constant(a_bar,lam_ladder[index_s]) # this is always positive
        integral = 0.5 * (score_tau - score_s) * (x_tau - x_s) * tsr_diff
        log_ratio = torch.clamp(integral.sum() , max=0.0)
        
        accept = torch.exp(log_ratio)
        accept_bool = (torch.rand_like(accept) < accept).float() 
        print(f"Time {t_val} {lam_ladder[index_t]:.2f} and {lam_ladder[index_s]:.2f} log_ratio {log_ratio.mean().item():.2f} accept {accept.mean().item():.2f} accept {accept_bool}")

        x_out[sl] = accept_bool * x_s + (1 - accept_bool) * x_tau
        x_out[ss] = accept_bool * x_tau + (1 - accept_bool) * x_s
        
    return x_out