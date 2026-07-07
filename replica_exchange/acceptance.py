# import torch

# @torch.no_grad()
# def _score_constant(a_bar, tsr_lam):
#     return 1 / (a_bar * tsr_lam + (1 - a_bar))

# @torch.no_grad()
# def _lam_ladder(lam_start, lam_end, n_replicas, device=None, dtype=None):
#     lam_ladder= torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)
#     print(lam_ladder)
#     return lam_ladder

# @torch.no_grad()
# def scale(grad, a_bar, lam_assignment):
#     """lam_assignment: (n_replicas,) — current lambda held by each physical slot."""
#     n_replicas = lam_assignment.shape[0]
#     lam_scalar = _score_constant(a_bar, lam_assignment)         # (n_replicas,)

#     samples_per_replica = grad.shape[0] // n_replicas
#     lam_scalar = lam_scalar.repeat_interleave(samples_per_replica)
#     lam_scalar = lam_scalar.view(-1, *[1] * (grad.dim() - 1))
#     return grad * lam_scalar

# @torch.no_grad()
# def swap(x_ladder, t_val, a_bar, lam_assignment, compute_eps, i=None, flow=False):
#     n_replicas = lam_assignment.shape[0]

#     sorted_slots = torch.argsort(lam_assignment)  # physical slot indices, ordered by current lambda ascending

#     if i is not None:
#         offset = i % 2
#     else:
#         offset = t_val % 2

#     pairs = [
#         (sorted_slots[k].item(), sorted_slots[k + 1].item())
#         for k in range(offset, n_replicas - 1, 2)
#     ]
#     index = x_ladder.shape[0] // n_replicas

#     interps = []
#     for index_t, index_s in pairs:
#         x_tau = x_ladder[index * index_t : index * (index_t + 1)]
#         x_s   = x_ladder[index * index_s : index * (index_s + 1)]
#         interps.append((x_tau + x_s) / 2)

#     x_ladder_ext = torch.cat([x_ladder] + interps, dim=0) if interps else x_ladder

#     v_ladder = compute_eps(x_ladder_ext, t_val)   # single forward pass, always
#     lam_assignment = lam_assignment.clone()

#     if flow:
#         # convert v-prediction -> eps for the acceptance math (SD3 / flow-matching models)
#         sigma_t = 1 - a_bar
#         x0_hat_ladder = x_ladder_ext - sigma_t * v_ladder
#         score_ladder = (x_ladder_ext - (1 - sigma_t) * x0_hat_ladder) / sigma_t
#     else:
#         # non-flow models predict eps directly
#         score_ladder = v_ladder

#     for i_pair, (index_t, index_s) in enumerate(pairs):
#         x_tau = x_ladder[index * index_t : index * (index_t + 1)]
#         x_s   = x_ladder[index * index_s : index * (index_s + 1)]

#         score_tau = score_ladder[index * index_t : index * (index_t + 1)]
#         score_s   = score_ladder[index * index_s : index * (index_s + 1)]
#         score_interp = score_ladder[index * i_pair + n_replicas * index : index * (i_pair + 1) + n_replicas * index]

#         lam_t = lam_assignment[index_t].clone()
#         lam_s = lam_assignment[index_s].clone()

#         tsr_diff = _score_constant(a_bar, lam_t) - _score_constant(a_bar, lam_s)
#         integral = (score_s + 4 * score_interp + score_tau) * (x_tau - x_s) * tsr_diff / 6

#         log_acceptance = torch.clamp(integral.sum(), max=0)
#         u = torch.rand_like(log_acceptance)
#         accept = (torch.log(u) < log_acceptance).float()

#         new_lam_t = accept * lam_s + (1 - accept) * lam_t
#         new_lam_s = accept * lam_t + (1 - accept) * lam_s
#         lam_assignment[index_t] = new_lam_t
#         lam_assignment[index_s] = new_lam_s

#         print(f"t {t_val} | lam {lam_t.item():.2f}↔{lam_s.item():.2f} "
#               f"| integral {integral.mean().item():.3f} | acceptance {accept.mean().item():.3f}")

#     return x_ladder, v_ladder[:n_replicas * index], lam_assignment

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

def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas, compute_eps, i=None, flow=None):
    x_out = x_ladder.clone()

    if i is not None:
        offset = i % 2
    else:
        offset = t_val % 2
    pairs = [(i_tau, i_tau + 1) for i_tau in range(offset, n_replicas - 1, 2)]
    index = x_ladder.shape[0] // n_replicas

    interps = []
    for i_pair, (index_t, index_s) in enumerate(pairs):
        x_tau = x_ladder[index * index_t : index * (index_t + 1)]
        x_s   = x_ladder[index * index_s : index * (index_s + 1)]
        interps.append((x_tau + x_s) / 2)

    if interps:
        x_ladder = torch.cat([x_ladder] + interps, dim=0)

    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=x_ladder.device, dtype=x_ladder.dtype)
    v_ladder = compute_eps(x_ladder, t_val)   # <-- plain, no grad tracking

    if flow:
        # convert v-prediction -> eps for the acceptance math (SD3 / flow-matching models)
        sigma_t = 1 - a_bar
        x0_hat_ladder = x_ladder - sigma_t * v_ladder
        score_ladder = (x_ladder - (1 - sigma_t) * x0_hat_ladder) / sigma_t
    else:
        # non-flow models predict eps directly
        score_ladder = - v_ladder / (1-a_bar)**0.5

    v_out = v_ladder.clone()

    for i_pair, (index_t, index_s) in enumerate(pairs):
        x_tau = x_ladder[index * index_t : index * (index_t + 1)]
        x_s   = x_ladder[index * index_s : index * (index_s + 1)]

        score_tau = score_ladder[index * index_t : index * (index_t + 1)]
        score_s   = score_ladder[index * index_s : index * (index_s + 1)]
        score_interp = score_ladder[index * i_pair + n_replicas * index : index * (i_pair + 1) + n_replicas * index]

        v_tau = v_ladder[index * index_t : index * (index_t + 1)]
        v_s   = v_ladder[index * index_s : index * (index_s + 1)]

        tsr_diff = _score_constant(a_bar, lam_ladder[index_t]) - _score_constant(a_bar, lam_ladder[index_s])
        integral = (score_s + 4 * score_interp + score_tau) * (x_tau - x_s) * tsr_diff / 6

        log_acceptance = torch.clamp(integral.sum(), max=0)
        u = torch.rand_like(log_acceptance)
        accept = (torch.log(u) < log_acceptance).float()
        accept = accept.view(-1, *[1] * (x_out.dim() - 1))

        x_out[index * index_t : index * (index_t + 1)] = accept * x_s   + (1 - accept) * x_tau
        x_out[index * index_s : index * (index_s + 1)] = accept * x_tau + (1 - accept) * x_s

        v_out[index * index_t : index * (index_t + 1)] = accept * v_s   + (1 - accept) * v_tau
        v_out[index * index_s : index * (index_s + 1)] = accept * v_tau + (1 - accept) * v_s

        print(f"t {t_val} | lam {lam_ladder[index_t]:.2f}↔{lam_ladder[index_s]:.2f} "
                f"| integral {integral.mean():.3f} | acceptance {accept.mean():.3f}")

    return x_out[:n_replicas*index], v_out[:n_replicas*index]