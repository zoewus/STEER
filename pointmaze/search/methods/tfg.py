from search.methods.base_guidance import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
from diffuser.models.helpers import apply_conditioning
import math
from torch.autograd import grad
import torch
from functools import partial

from search.utils import rescale_grad

#acceptance
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from functools import partial
from replica_exchange.acceptance import swap, scale, AdaptiveLadder

class TFGGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(TFGGuidance, self).__init__(args, **kwargs)
        self.ladder = None   # lazy-init in guide_step, once we have a tensor to read device/dtype from

    def _maybe_init_ladder(self, x, i):
        if self.ladder is None or i == 0:
            self.ladder = AdaptiveLadder(
                lam_start=self.args.lam_start,
                lam_end=self.args.lam_end,
                n_replicas=self.args.n_particles,
                target_accept=getattr(self.args, "target_accept", 0.3),
                kappa0=getattr(self.args, "ladder_kappa0", 1.0),
                decay=getattr(self.args, "ladder_decay", 0.6),
                device=x.device,
                dtype=x.dtype,
            )

    @torch.enable_grad()
    def tilde_get_guidance(self, x0, mc_eps, return_logp=False, **kwargs):

        # flat_x0 = (x0[None] + mc_eps) #.reshape(-1, *x0.shape[1:])
        # v_func = torch.vmap(partial(self.guider.get_guidance,
        #                             return_logp=True, 
        #                             check_grad=False,
        #                             **kwargs))
        # outs = v_func(flat_x0)
        
        # avg_logprobs = torch.logsumexp(outs, dim=0) - math.log(mc_eps.shape[0])
        
        flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
        outs = self.guider.get_guidance(flat_x0, return_logp=True, check_grad=False, **kwargs)

        avg_logprobs = torch.logsumexp(outs.reshape(mc_eps.shape[0], x0.shape[0]), dim=0) - math.log(mc_eps.shape[0])
        
        if return_logp:
            return avg_logprobs

        _grad = torch.autograd.grad(avg_logprobs.sum(), x0)[0]
        _grad = rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad
    
    def get_noise(self, std,alpha_prod_t, shape, eps_bsz=4, **kwargs):
        # if std == 0.0:
        #     return torch.zeros((1, *shape), device=self.device)
        return torch.stack([self.noise_fn(torch.zeros(shape, device=self.device), std, alpha_prod_t, **kwargs) for _ in range(eps_bsz)]) 
    # randn_tensor((4, *shape), device=self.device, generator=self.generator) * std
    
    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_mu(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.mu_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.mu *  scheduler[t] * len(scheduler) / scheduler.sum()
    
    def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.sigma_schedule == 'decrease':    # beta_t
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif self.args.sigma_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.sigma *  scheduler[t]

    def guide_step(
        self,
        x: torch.Tensor,
        i: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:
        self._maybe_init_ladder(x, i)
        cond = kwargs.get("cond", None)

        t = ts[i]   # convert from int space to tensor space
        batched_t = t.repeat(x.shape[0])
        alpha_prod_t = alpha_prod_ts[i]
        alpha_prod_t_prev = alpha_prod_t_prevs[i]

        rho = self.get_rho(i, alpha_prod_ts, alpha_prod_t_prevs)
        mu = self.get_mu(i, alpha_prod_ts, alpha_prod_t_prevs)
        std = self.get_std(i, alpha_prod_ts, alpha_prod_t_prevs)

        for recur_step in range(self.args.recur_steps):

            # sample noise to estimate the \tilde p distribution
            mc_eps = self.get_noise(std, alpha_prod_t, x.shape, self.args.eps_bsz, **kwargs)

            # Compute guidance on x_t, and obtain Delta_t
            if self.args.rho != 0.0:
                with torch.enable_grad():
                    x_g = x.clone().detach().requires_grad_()
                    unet_output = unet(x_g, cond, batched_t)
                    new_epsilon = (
                        (x_g - alpha_prod_t ** (0.5) * unet_output) / (1 - alpha_prod_t) ** (0.5)
                    )
                    x = x_g.clone()
                    new_epsilon_scaled = scale(new_epsilon, alpha_prod_t, self.ladder.lam_ladder)
                    
                    # invert x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps  =>  x0 = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
                    x0_tempered = (
                        (x - (1 - alpha_prod_t) ** (0.5) * new_epsilon_scaled) / alpha_prod_t ** (0.5)
                    )
                    x0 = self._predict_x0(x_g, x0_tempered, alpha_prod_t, **kwargs)
                    x0 = apply_conditioning(x0, cond, 2) ## debug
                    logprobs = self.tilde_get_guidance(
                        x0, mc_eps, return_logp=True, **kwargs)
                    Delta_t = grad(logprobs.sum(), x_g)[0]
                    Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
                    Delta_t = Delta_t * rho
                    
            else:
                Delta_t = torch.zeros_like(x)
                x0 = self._predict_x0(x, unet(x, cond, batched_t), alpha_prod_t, **kwargs)

            # Compute guidance on x_{0|t}
            new_x0 = x0.clone().detach()
            
            for _ in range(self.args.iter_steps):
                if self.args.mu != 0.0:
                    new_x0 += mu * self.tilde_get_guidance(
                        new_x0.detach().requires_grad_(), mc_eps, **kwargs)
            Delta_0 = new_x0 - x0
            
            # predict x_{t-1} using S(zt, hat_epsilon, t), this is also DDIM sampling
            alpha_t = alpha_prod_t / alpha_prod_t_prev
            x_prev = self._predict_x_prev_from_zero(
                x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
            if t > 0 or recur_step < self.args.recur_steps - 1:
                x_prev += Delta_t / alpha_t ** 0.5 + Delta_0 * alpha_prod_t_prev ** 0.5
            x_prev = apply_conditioning(x_prev, cond, 2)            
            x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs).detach().requires_grad_(False)
            
            x = apply_conditioning(x, cond, 2)

        if self.args.replica_exchange and i < (len(ts) -1 ):
            x_prev, _, accept_per_pair = swap(x_prev, t, alpha_prod_t, self.ladder.lam_ladder,
                                            new_epsilon, i=i, flow=None)
            self.ladder.update(accept_per_pair)
        return x_prev, {"x0": x0, "logprobs": logprobs}