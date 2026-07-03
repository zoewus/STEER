import torch

@torch.no_grad()
def _score_constant(a_bar, tsr_lam):
    output = 1 / (a_bar * tsr_lam + (1 - a_bar))
    return output

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device=None):
    lam_ladder = torch.linspace(lam_start, lam_end, n_replicas, device=device)
    return lam_ladder

@torch.no_grad()
def scale(grad, a_bar, lam_start, lam_end, n_replicas):
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, grad.device)          # (n_replicas,)
    lam_ladder_t = _score_constant(a_bar, lam_ladder)                  # (n_replicas,) vectorized

    samples_per_replica = grad.shape[0] // n_replicas
    lam_ladder_t = lam_ladder_t.repeat_interleave(samples_per_replica)  # (n_replicas * samples_per_replica,)

    lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
    return grad * lam_ladder_t

def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas, compute_eps):

    offset = t_val % 2
    pairs = [(i, i + 1) for i in range(offset, n_replicas - 1, 2)]
    index = x_ladder.shape[0] // n_replicas

    for i, (index_t, index_s) in enumerate(pairs):
        x_tau = x_ladder[index * index_t : index * (index_t + 1)].clone()
        x_s   = x_ladder[index * index_s : index * (index_s + 1)].clone()
        x_interp =(x_tau + x_s)/2
        x_ladder = torch.cat([x_ladder, x_interp], dim = 0)

    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, x_ladder.device) 
    score_ladder = compute_eps(x_ladder, t_val)

    for i, (index_t, index_s) in enumerate(pairs):
        x_tau = x_ladder[index * index_t : index * (index_t + 1)].clone()
        x_s   = x_ladder[index * index_s : index * (index_s + 1)].clone()

        score_tau = score_ladder[index * index_t : index * (index_t + 1)].clone()
        score_s = score_ladder[index * index_s : index * (index_s + 1)].clone()
        score_interp = score_ladder[index * i + n_replicas * index: index * (i + 1) + n_replicas * index].clone()

        tsr_diff = _score_constant(a_bar, lam_ladder[index_t]) - _score_constant(a_bar, lam_ladder[index_s])
        integral =  (score_s + 4 * score_interp + score_tau) * (x_tau - x_s) * tsr_diff / 6

        log_acceptance = torch.clamp(integral.sum(), max=0)  # assume this is bs = 1
        u = torch.rand_like(log_acceptance)
        accept = (torch.log(u) < log_acceptance).float()  # shape (batch,)

        x_ladder[index * index_t : index * (index_t + 1)] = accept * x_s   + (1 - accept) * x_tau
        x_ladder[index * index_s : index * (index_s + 1)] = accept * x_tau + (1 - accept) * x_s

        score_ladder[index * index_t : index * (index_t + 1)] = accept * score_s   + (1 - accept) * score_tau
        score_ladder[index * index_s : index * (index_s + 1)] = accept * score_tau + (1 - accept) * score_s

        print(f"t {t_val} | lam {lam_ladder[index_t]:.2f}↔{lam_ladder[index_s]:.2f} "
              f"| integral {integral.mean():.3f} | acceptance {accept.mean():.3f}")

    return x_ladder[:n_replicas*index], score_ladder[:n_replicas*index]