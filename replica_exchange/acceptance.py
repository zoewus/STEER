import torch

@torch.no_grad()
def _score_constant(a_bar, tsr_lam, eps=1e-6):
    denom = a_bar * tsr_lam + (1 - a_bar)
    denom = denom.clamp_min(eps) if torch.is_tensor(denom) else max(denom, eps)
    return 1 / denom

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device=None, dtype=None):
    lam_ladder = torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)
    return lam_ladder

class AdaptiveLadder:
    """
    Feedback-optimized ladder (Katzgraber et al. / Vousden et al. style).
    Endpoints (lam_start, lam_end) stay fixed; the gaps between rungs are
    reshaped so every adjacent pair's acceptance converges toward each
    other (equalization). `target_accept` seeds the initial EMA and gives
    the *direction* of each correction; whether that absolute level is
    reachable depends on n_replicas and the total (lam_start, lam_end)
    range -- if the equalized level ends up far from your target band,
    adjust those instead of expecting this loop to do it alone.
    Adaptation strength decays ~ 1/step**decay so the ladder freezes over
    time (needed so swaps still satisfy detailed balance asymptotically).
    """
    def __init__(self, lam_start, lam_end, n_replicas, target_accept=0.3,
                 kappa0=1.0, decay=0.6, ema_beta=0.9, device=None, dtype=None):
        self.lam_start = lam_start
        self.lam_end = lam_end
        self.target_accept = target_accept
        self.kappa0 = kappa0
        self.decay = decay
        self.ema_beta = ema_beta
        self.step = 0
        self.lam_ladder = torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)
        self.accept_ema = torch.full((n_replicas - 1,), target_accept, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, accept_per_pair):
        """accept_per_pair: shape (n_replicas-1,). Use NaN for any pair not
        attempted this call (e.g. the checkerboard offset skips half the
        pairs each swap step) -- those entries are left untouched."""
        self.step += 1
        kappa = self.kappa0 / (self.step ** self.decay)

        mask = ~torch.isnan(accept_per_pair)
        self.accept_ema[mask] = (
            self.ema_beta * self.accept_ema[mask] + (1 - self.ema_beta) * accept_per_pair[mask]
        )

        gaps = torch.diff(self.lam_ladder)
        log_ratio = torch.log(self.accept_ema.clamp_min(1e-3) / self.target_accept)
        new_gaps = (gaps * torch.exp(kappa * log_ratio)).clamp_min(1e-4)
        new_gaps = new_gaps * (self.lam_end - self.lam_start) / new_gaps.sum()  # keep endpoints fixed

        self.lam_ladder = torch.cat([
            torch.tensor([self.lam_start], dtype=new_gaps.dtype, device=new_gaps.device),
            self.lam_start + torch.cumsum(new_gaps, dim=0),
        ])


def scale(grad, a_bar, lam_ladder):
    """lam_ladder is now passed in directly (from AdaptiveLadder.lam_ladder)
    instead of being reconstructed from lam_start/lam_end/n_replicas every call."""
    lam_ladder_t = _score_constant(a_bar, lam_ladder)
    lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
    return grad * lam_ladder_t


def swap(x_ladder, t_val, a_bar, lam_ladder, eps_ladder, i=None, flow=None):
    x_out = x_ladder.clone()
    eps_out = eps_ladder.clone()
    n_replicas = lam_ladder.shape[0]

    offset = (i % 2) if i is not None else (t_val % 2)
    pairs = [(i_tau, i_tau + 1) for i_tau in range(offset, n_replicas - 1, 2)]
    index = x_ladder.shape[0] // n_replicas

    if flow:
        sigma_t = 1 - a_bar
        x0_hat_ladder = x_ladder - sigma_t * eps_ladder
        score_ladder = (x_ladder - (1 - sigma_t) * x0_hat_ladder) / sigma_t
    else:
        score_ladder = -eps_ladder / (1 - a_bar) ** 0.5

    # NaN for pairs not attempted this call -- AdaptiveLadder.update masks these out
    accept_per_pair = torch.full((n_replicas - 1,), float('nan'), dtype=x_ladder.dtype, device=x_ladder.device)

    for i_pair, (index_t, index_s) in enumerate(pairs):
        x_tau = x_ladder[index * index_t: index * (index_t + 1)]
        x_s = x_ladder[index * index_s: index * (index_s + 1)]

        score_tau = score_ladder[index * index_t: index * (index_t + 1)]
        score_s = score_ladder[index * index_s: index * (index_s + 1)]

        eps_tau = eps_ladder[index * index_t: index * (index_t + 1)]
        eps_s = eps_ladder[index * index_s: index * (index_s + 1)]

        tsr_diff = _score_constant(a_bar, lam_ladder[index_t]) - _score_constant(a_bar, lam_ladder[index_s])
        integral = (score_s + score_tau) * (x_tau - x_s) * tsr_diff / 2

        integral_per_sample = integral.flatten(1).sum(dim=1)
        log_acceptance = torch.clamp(integral_per_sample, max=0)
        u = torch.rand_like(log_acceptance)
        accept = (torch.log(u) < log_acceptance).float()

        x_out[index * index_t: index * (index_t + 1)] = accept * x_s + (1 - accept) * x_tau
        x_out[index * index_s: index * (index_s + 1)] = accept * x_tau + (1 - accept) * x_s

        eps_out[index * index_t: index * (index_t + 1)] = accept * eps_s + (1 - accept) * eps_tau
        eps_out[index * index_s: index * (index_s + 1)] = accept * eps_tau + (1 - accept) * eps_s

        accept_per_pair[index_t] = accept.mean()  # pair index == lower rung index

        print(f"t {t_val} | lam {lam_ladder[index_t]:.3f}<->{lam_ladder[index_s]:.3f} "
              f"| integral {integral.mean():.3f} | acceptance {accept.mean():.3f}")

    return x_out, eps_out, accept_per_pair

# import torch

# @torch.no_grad()
# def _score_constant(a_bar, tsr_lam):
#     output = 1 / (a_bar * tsr_lam +(1-a_bar))
#     return output

# @torch.no_grad()
# def _lam_ladder(lam_start, lam_end, n_replicas, device=None, dtype=None):
#     lam_ladder = torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)
#     return lam_ladder

# def scale(grad, a_bar, lam_start, lam_end, n_replicas):
#     lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)
#     lam_ladder_t = _score_constant(a_bar, lam_ladder)                  # (n_replicas,) vectorized
    
#     lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
#     return grad * lam_ladder_t

# def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas, eps_ladder, i=None, flow=None):
#     x_out = x_ladder.clone()
#     eps_out = eps_ladder.clone()

#     if i is not None:
#         offset = i % 2
#     else:
#         offset = t_val % 2
#     pairs = [(i_tau, i_tau + 1) for i_tau in range(offset, n_replicas - 1, 2)]
#     index = x_ladder.shape[0] // n_replicas

#     # interps = []
#     # for i_pair, (index_t, index_s) in enumerate(pairs):
#     #     x_tau = x_ladder[index * index_t : index * (index_t + 1)]
#     #     x_s   = x_ladder[index * index_s : index * (index_s + 1)]
#     #     interps.append((x_tau + x_s) / 2)
    
#     # x_interps = torch.cat(interps, dim=0)
#     #eps_interps = compute_eps(x_interps, t_val)

#     lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=x_ladder.device, dtype=x_ladder.dtype)

#     if flow:
#         # convert v-prediction -> eps for the acceptance math (SD3 / flow-matching models)
#         sigma_t = 1 - a_bar
#         x0_hat_ladder = x_ladder - sigma_t * eps_ladder
#         score_ladder = (x_ladder - (1 - sigma_t) * x0_hat_ladder) / sigma_t
#     else:
#         # non-flow models predict eps directly
#         score_ladder = - eps_ladder / (1 - a_bar) ** 0.5
#         # score_interps = - eps_interps / (1 - a_bar) ** 0.5

#     for i_pair, (index_t, index_s) in enumerate(pairs):
#         x_tau = x_ladder[index * index_t : index * (index_t + 1)]
#         x_s   = x_ladder[index * index_s : index * (index_s + 1)]

#         score_tau = score_ladder[index * index_t : index * (index_t + 1)]
#         score_s   = score_ladder[index * index_s : index * (index_s + 1)]
#         # score_interp = score_interps[index * i_pair : index * (i_pair + 1)]

#         eps_tau = eps_ladder[index * index_t : index * (index_t + 1)]
#         eps_s   = eps_ladder[index * index_s : index * (index_s + 1)]

#         tsr_diff = _score_constant(a_bar, lam_ladder[index_t]) - _score_constant(a_bar, lam_ladder[index_s])
#         integral = - (score_s + score_tau) * (x_tau - x_s) * tsr_diff / 2

#         integral_per_sample = integral.flatten(1).sum(dim=1)          # shape: (index,)
#         log_acceptance = torch.clamp(integral_per_sample, max=0)      # shape: (index,)
#         u = torch.rand_like(log_acceptance)
#         accept = (torch.log(u) < log_acceptance).float()   

#         x_out[index * index_t : index * (index_t + 1)] = accept * x_s   + (1 - accept) * x_tau
#         x_out[index * index_s : index * (index_s + 1)] = accept * x_tau + (1 - accept) * x_s

#         eps_out[index * index_t : index * (index_t + 1)] = accept * eps_s   + (1 - accept) * eps_tau
#         eps_out[index * index_s : index * (index_s + 1)] = accept * eps_tau + (1 - accept) * eps_s

#         print(f"t {t_val} | lam {lam_ladder[index_t]:.2f}↔{lam_ladder[index_s]:.2f} "
#                 f"| integral {integral.mean():.3f} | acceptance {accept.mean():.3f}")

#     return x_out, eps_out