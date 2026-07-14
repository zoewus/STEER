import torch

@torch.no_grad()
def _score_constant(a_bar, tsr_lam):
    return 1 / (a_bar * tsr_lam + (1 - a_bar))

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device, dtype):
    lam_ladder = torch.round(torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype), decimals=5)
    return lam_ladder


def init_temp_idx(n_replicas, device):
    """
    Call ONCE before sampling starts (e.g. in __init__ or lazily on first
    guide_step call). temp_idx[slot] = index into lam_ladder that is
    CURRENTLY assigned to that slot/walker. Starts as identity and gets
    permuted in place by swap_temperatures() as sampling proceeds.
    """
    return torch.arange(n_replicas, device=device)


def scale(grad, a_bar, lam_start, lam_end, n_replicas, temp_idx=None):
    """
    Same as original `scale`, but if temp_idx is provided, each slot k is
    tempered by lam_ladder[temp_idx[k]] (its currently-assigned lambda)
    instead of lam_ladder[k] directly. Pass temp_idx=None to recover the
    original sample-swap behavior exactly.
    """
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)

    lam_per_slot = lam_ladder[temp_idx] if temp_idx is not None else lam_ladder

    lam_ladder_t = _score_constant(a_bar, lam_per_slot)
    lam_ladder_t = lam_ladder_t.view(-1, *[1] * (grad.dim() - 1))
    return grad * lam_ladder_t


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
    step_val = i
    pairs = []

    if i % 2 == 0:
        offset = (step_val % 4) // 2
        for i_tau in range(offset, n_replicas - 1, 2):
            index_t = get_slot_for_lambda(temp_idx, i_tau)
            index_s = get_slot_for_lambda(temp_idx, i_tau+1)
            pairs.append((index_t, index_s))

    index = x_ladder.shape[0] // n_replicas
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=x_ladder.device, dtype=x_ladder.dtype)

    score_ladder = -eps_ladder / (1 - a_bar) ** 0.5

    for index_t, index_s in pairs:
        sl = slice(index * index_t, index * (index_t + 1))
        ss = slice(index * index_s, index * (index_s + 1))

        x_tau, x_s = x_ladder[sl], x_ladder[ss]
        score_tau, score_s = score_ladder[sl], score_ladder[ss]

        lam_t_val = lam_ladder[temp_idx[index_t]]
        lam_s_val = lam_ladder[temp_idx[index_s]]


        tsr_diff = (1/lam_t_val) - (1/lam_s_val)
        integral = - (score_tau + score_s) * (x_tau - x_s) #* tsr_diff
        log_ratio = torch.clamp(integral.mean(), max=0.0)

        accept = torch.exp(log_ratio)
        accept_bool = (torch.rand(accept.shape, dtype=accept.dtype, device=accept.device) < accept).bool()

        print(f"Time {t_val} i {i} lam_t {lam_t_val:.5f} lam_s {lam_s_val:.5f} "
              f"log_ratio {log_ratio.item():.4f} accept {accept.item():.4f} "
              f"swapped {accept_bool.item()}")

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