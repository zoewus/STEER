import torch
import math

@torch.no_grad()
def _score_constant(a_bar, tsr_lam):
    return 1 / (a_bar * tsr_lam + (1 - a_bar))

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device, dtype, spacing="linear"):
    if n_replicas ==1:
        return torch.tensor([lam_start], device=device, dtype=dtype)
    
    if spacing == "linear":
        lam_ladder = torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)

    elif spacing == "geometric":
        # cluster points closer to lam_end (e.g. 1.0)
        # gap = lam_end - lam, shrinks geometrically from (lam_end - lam_start) down to ~0
        gap_start = lam_end - lam_start
        gap_end = gap_start * 1e-3  # smallest gap fraction; tune as needed
        gaps = torch.logspace(
            math.log10(gap_start), math.log10(gap_end), n_replicas,
            device=device, dtype=dtype
        )
        lam_ladder = lam_end - gaps
        lam_ladder[0] = lam_start   # ensure exact endpoints
        lam_ladder[-1] = lam_end

    else:
        raise ValueError(f"unknown spacing: {spacing}")

    return lam_ladder

def init_temp_idx(n_replicas, device):
    return torch.arange(n_replicas, device=device)

def scale(grad, a_bar, lam_start, lam_end, n_replicas, temp_idx=None):
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)
    lam_per_slot = lam_ladder[temp_idx] if temp_idx is not None else lam_ladder
    lam_ladder_t = _score_constant(a_bar, lam_per_slot)
    lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
    return lam_ladder_t


def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas,
                       eps_ladder, temp_idx, i=None, flow=None):
    """
    Same acceptance criterion / math as the original `swap`, but on accept we
    swap the TEMPERATURE LABELS (temp_idx) instead of moving data between
    slots. x_ladder is returned unchanged; temp_idx is mutated in place and
    also returned for convenience/clarity at the call site.

    temp_idx must be created once via init_temp_idx() and threaded through
    every guide_step call (persisted on self, not recreated each step).
    """
    if i is not None:
        step_val = i
    else:
        step_val = t_val
        
    offset = step_val % 2  # 0 or 1
    pairs = []
    for i_tau in range(offset, n_replicas - 1, 2):
        index_t = get_slot_for_lambda(temp_idx, i_tau)
        index_s = get_slot_for_lambda(temp_idx, i_tau + 1)
        pairs.append((index_t, index_s))

    index = x_ladder.shape[0] // n_replicas
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=x_ladder.device, dtype=x_ladder.dtype)

    score_ladder = - eps_ladder / (1 - a_bar)

    for index_t, index_s in pairs:
        sl = slice(index * index_t, index * (index_t + 1))
        ss = slice(index * index_s, index * (index_s + 1))

        x_tau, x_s = x_ladder[sl], x_ladder[ss]
        score_tau, score_s = score_ladder[sl], score_ladder[ss]

        lam_t_val = lam_ladder[temp_idx[index_t]]
        lam_s_val = lam_ladder[temp_idx[index_s]]

        tsr_diff = _score_constant(a_bar, lam_t_val) - _score_constant(a_bar, lam_s_val)
        integral = - 0.5 *(score_tau + score_s) * (x_tau - x_s) * tsr_diff
        log_ratio = torch.clamp(integral.sum() , max=0.0)
        accept = torch.exp(log_ratio)
        accept_bool = (torch.rand(accept.shape, dtype=accept.dtype, device=accept.device) < accept).bool()

        print(
            f"i={i} pair=({index_t},{index_s}) "
            f"lam_t={lam_t_val.item():.5f} lam_s={lam_s_val.item():.5f} "
            f"tsr_diff={tsr_diff.item():.6f} "
            f"integral_mean={integral.mean().item():.6f} integral_std={integral.std().item():.6f} "
            f"log_ratio={log_ratio.item():.4f} accept={accept.item():.4f} "
            f"swapped={bool(accept_bool.item())}"
        )

        if accept_bool:
            # Swap the TEMPERATURE LABELS, not the data.
            tmp = temp_idx[index_t].clone()
            temp_idx[index_t] = temp_idx[index_s]
            temp_idx[index_s] = tmp

    return temp_idx


def get_slot_for_lambda(temp_idx, target_lam_index):
    """
    At readout time, find which slot currently holds the walker running at
    the given lambda-ladder index (e.g. the index closest to lam=1.0 -- the
    'coldest' / least-tempered replica, which is normally what you want as
    your final output sample).
    """
    match = (temp_idx == target_lam_index).nonzero(as_tuple=True)[0]
    assert match.numel() == 1, f"expected exactly one slot for lambda index {target_lam_index}, got {match.numel()}"
    return match.item()