import torch
import numpy as np

@torch.no_grad()
def _score_constant(a_bar, tsr_lam):
    return 1 / (a_bar * tsr_lam + (1 - a_bar))

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device, dtype):
    """
    Build a lambda ladder centered on 1.0 (in log-space), spanning
    [lam_start, 1/lam_start]. Still 1-D, shape (n_replicas,) — this is just
    the fixed set of rung values, shared by every element of the event grid.
    """
    # if lam_start == 1.0:
    #     return torch.ones(n_replicas, device=device, dtype=dtype)

    # lam_end = 1.0 / lam_start
    # lo, hi = min(lam_start, lam_end), max(lam_start, lam_end)

    # lam_ladder = torch.tensor(
    #     np.geomspace(lo, hi, n_replicas), device=device, dtype=dtype
    # )
    n_pairs = (n_replicas + 1) // 2  # handles odd n_replicas by dropping the leftover 1/λ
    step = 0.01

    lam_values = lam_start + step * torch.arange(n_pairs, device=device, dtype=dtype)
    lam_ladder = torch.stack([lam_values, 1.0 / lam_values], dim=1).flatten()[:n_replicas]
    return lam_ladder


def init_temp_idx(n_replicas, event_shape, device):
    """
    Call ONCE before sampling starts. temp_idx now has shape
    (n_replicas, *event_shape) — e.g. (n_replicas, 600, 4), matching x.

    temp_idx[k, i, j] = the lambda-ladder rung CURRENTLY assigned to slot k,
    for grid position (i, j). Every position (i, j) runs its own independent
    permutation of the n_replicas rungs across slots. Starts as identity
    (broadcast across the event grid) and gets permuted in place per-position
    by swap().
    """
    base = torch.arange(n_replicas, device=device).view(n_replicas, *[1] * len(event_shape))
    return base.expand(n_replicas, *event_shape).clone()


def scale(grad, a_bar, lam_start, lam_end, n_replicas, temp_idx=None):
    """
    Same as before. If temp_idx has shape (n_replicas, *event_shape) matching
    grad, lam_ladder[temp_idx] already broadcasts elementwise against grad —
    no reshape needed. Pass temp_idx=None to recover the original
    one-lambda-per-slot behavior (broadcast over the event dims).
    """
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)

    if temp_idx is not None:
        lam_per_elem = lam_ladder[temp_idx]  # shape == temp_idx.shape == grad.shape
    else:
        lam_per_elem = lam_ladder.view(-1, *[1] * (grad.dim() - 1))

    tsr_t = _score_constant(a_bar, lam_per_elem)
    return grad * tsr_t


def get_slot_for_lambda(temp_idx, target_lam_index):
    """
    Elementwise version: for each event-grid position, find which slot
    currently holds the walker running at `target_lam_index`.

    Returns a tensor of shape event_shape (temp_idx.shape[1:]), dtype long,
    giving the slot index per position.
    """
    match = (temp_idx == target_lam_index)  # (n_replicas, *event_shape)
    counts = match.sum(dim=0)
    assert torch.all(counts == 1), (
        f"expected exactly one slot per position for lambda index {target_lam_index}, "
        f"got counts ranging {counts.min().item()}-{counts.max().item()}"
    )
    return match.long().argmax(dim=0)  # (*event_shape,)


def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas,
         eps_ladder, unet, cond, temp_idx, i=None, flow=None):
    """
    Elementwise replica-exchange. temp_idx has shape (n_replicas, *event_shape)
    matching x_ladder, so every grid position (e.g. each of the 600*4
    positions) does its own independent accept/reject and its own swap of
    rung labels. x_ladder is returned unchanged; temp_idx is mutated in place.
    """

    device = x_ladder.device
    event_shape = temp_idx.shape[1:]
    idx_grid = torch.meshgrid(
        *[torch.arange(s, device=device) for s in event_shape], indexing='ij'
    )  # tuple of index tensors, each shape event_shape

    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=device, dtype=x_ladder.dtype)
    score_ladder = -eps_ladder / (1 - a_bar) ** 0.5  # same shape as x_ladder
    
    offset = 0
    spacing = 2
    for i_tau in range(offset, n_replicas, spacing):
        if i_tau + spacing >= n_replicas:
            continue
        index_t = get_slot_for_lambda(temp_idx, i_tau)              # (*event_shape,)
        index_s = get_slot_for_lambda(temp_idx, i_tau + spacing)    # (*event_shape,)
        
        x_tau = x_ladder[(index_t, *idx_grid)]
        x_s = x_ladder[(index_s, *idx_grid)]
        score_tau = score_ladder[(index_t, *idx_grid)]
        score_s = score_ladder[(index_s, *idx_grid)]

        # These are scalars: temp_idx[index_t] == i_tau and temp_idx[index_s] == i_tau+1
        # by construction, so no need to re-index lam_ladder per position.
        lam_t_val = lam_ladder[i_tau]
        lam_s_val = lam_ladder[i_tau + 1]

        tsr_diff = _score_constant(a_bar, lam_t_val) - _score_constant(a_bar, lam_s_val)
        integral = -0.5 * (score_tau + score_s) * (x_tau - x_s) * tsr_diff
        log_ratio = torch.clamp(integral, max=0.0)

        accept = torch.exp(log_ratio)
        accept_bool = torch.rand(accept.shape, dtype=accept.dtype, device=device) < accept

        print(f"Time {t_val} i {i} lam_t {lam_t_val:.5f} lam_s {lam_s_val:.5f} "
              f"log_ratio {log_ratio.mean().item():.4f} accept {accept.mean().item():.4f} "
              f"swapped_frac {accept_bool.float().mean().item():.4f}")

        # Swap rung labels wherever accepted, independently per grid position.
        val_t = temp_idx[(index_t, *idx_grid)]  # == i_tau everywhere
        val_s = temp_idx[(index_s, *idx_grid)]  # == i_tau + 1 everywhere

        new_val_t = torch.where(accept_bool, val_s, val_t)
        new_val_s = torch.where(accept_bool, val_t, val_s)

        temp_idx[(index_t, *idx_grid)] = new_val_t
        temp_idx[(index_s, *idx_grid)] = new_val_s

    return temp_idx