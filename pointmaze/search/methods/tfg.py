#acceptance
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from acceptance import swap, scale


from search.methods.base_guidance import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor
from diffuser.models.helpers import apply_conditioning
import math
from torch.autograd import grad
import torch
from functools import partial

from search.utils import rescale_grad



class TFGGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(TFGGuidance, self).__init__(args, **kwargs)

    def _compute_eps(self, x_t, t_val, cond, unet, alpha_prod_t, std, rho, **kwargs):
        mc_eps = self.get_noise(std, x_t.shape, self.args.eps_bsz, **kwargs)
        batched_t = t_val.repeat(x_t.shape[0])
        K = x_t.shape[0]  # wherever your new interpolation dim landed
        v = cond[next(iter(cond))]
        pad_len = K - v.shape[0]
        cond = {key: torch.cat([v, v[0:1].repeat(pad_len, *([1] * (v.dim() - 1)))], dim=0) for key, v in cond.items()}
        # cond = {
        #     key: torch.cat([v, v[0:1]], dim=0)
        #     for key, v in cond.items()
        # }
        x_g = x_t.clone().detach().requires_grad_()
        x0 = self._predict_x0(x_g, unet(x_g, cond, batched_t), alpha_prod_t, **kwargs)
        x0 = apply_conditioning(x0, cond, 2) ## debug
        logprobs = self.tilde_get_guidance(
            x0, mc_eps, return_logp=True, **kwargs)
        Delta_t = grad(logprobs.sum(), x_g)[0]
        Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
        Delta_t = Delta_t * rho
        return Delta_t

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
    
    def get_noise(self, std, shape, eps_bsz=4, **kwargs):
        # if std == 0.0:
        #     return torch.zeros((1, *shape), device=self.device)
        return torch.stack([self.noise_fn(torch.zeros(shape, device=self.device), std, **kwargs) for _ in range(eps_bsz)]) 
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
            mc_eps = self.get_noise(std, x.shape, self.args.eps_bsz, **kwargs)

            # Compute guidance on x_t, and obtain Delta_t
            if self.args.rho != 0.0:
                with torch.enable_grad():
                    x_g = x.clone().detach().requires_grad_()
                    x0 = self._predict_x0(x_g, unet(x_g, cond, batched_t), alpha_prod_t, **kwargs)
                    x0 = apply_conditioning(x0, cond, 2) ## debug
                    logprobs = self.tilde_get_guidance(
                        x0, mc_eps, return_logp=True, **kwargs)
                    Delta_t = grad(logprobs.sum(), x_g)[0]
                    Delta_t = rescale_grad(Delta_t, clip_scale=self.args.clip_scale, **kwargs)
                    Delta_t = Delta_t * rho

                    Delta_t = scale(Delta_t, alpha_prod_t, self.args.lam_start, self.args.lam_end, self.args.n_particles)
                    
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
            
        if self.args.replica_exchange:
            
            compute_eps_fn = partial(
                self._compute_eps,
                unet=unet, alpha_prod_t=alpha_prod_t, std=std, rho=rho, **kwargs
            )

            x_prev, _ = swap(x_prev, t, alpha_prod_t, self.args.lam_start, self.args.lam_end, self.args.n_particles, compute_eps_fn)
        return x_prev, {"x0": x0, "logprobs": logprobs}



