from search.configs import Arguments
from search.base_policy import BasePolicy
from search.methods.base_guidance import BaseGuidance
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffuser.models.helpers import apply_conditioning

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),  "..", "..")))
from functools import partial
from replica_exchange.acceptance import swap, scale, init_temp_idx, _lam_ladder

class SearchPolicy(BasePolicy):
    def __init__(self, args:Arguments, **kwargs):
        super().__init__(args=args, **kwargs)
    
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        self.inference_steps = self.args.inference_steps
        self.eta = self.args.eta
        self.generator = torch.manual_seed(self.args.seed)
        self.per_sample_batch_size = self.args.per_sample_batch_size

        from search.utils import space_timesteps

        ts = space_timesteps(self.diffusion.n_timesteps, self.inference_steps) ## reversed order
        alpha_prod_ts = self.diffusion.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat((self.diffusion.alphas_cumprod[ts[1:]], torch.tensor([1.0], device=self.diffusion.alphas_cumprod.device, dtype=torch.float32)),dim=0)

        ## shape: [inference_steps]
        self.ts = torch.tensor(ts, device=self.device, dtype=torch.long)
        self.alpha_prod_ts = alpha_prod_ts
        self.alpha_prod_t_prevs = alpha_prod_t_prevs


    def sample(self, cond, guidance:BaseGuidance, **kwargs):

        x = randn_tensor((self.per_sample_batch_size, self.diffusion.horizon,self.diffusion.transition_dim), generator=self.generator, device=self.device)
        noise = torch.randn(
            x.shape,
            device=x.device,
            dtype=x.dtype,
        )
        lam_ladder_t = _lam_ladder(self.args.lam_start, self.args.lam_end, self.args.n_particles, x.device, x.dtype)
        lam_ladder_t = lam_ladder_t.view(-1, *[1] * (x.dim() - 1))
        x *=  torch.sqrt(lam_ladder_t)

        x = apply_conditioning(x, cond, self.diffusion.action_dim)

        temp_idx = init_temp_idx(self.args.n_particles,  x.device)

        guidance.reset()
        i = 0
        total_compute = 0
        while i < len(self.ts):
            total_compute += self.args.recur_steps * x.shape[0]   ## NFEs per step
            x, extras, temp_idx = guidance.guide_step(
                x,
                i,
                self.diffusion.model,
                self.ts,
                self.alpha_prod_ts,
                self.alpha_prod_t_prevs,
                self.eta,
                temp_idx,
                cond=cond,
                post_process=self.unnormalize,
            )

            i = extras.get("i_next", i + 1)
          
        return x, {'compute': total_compute}