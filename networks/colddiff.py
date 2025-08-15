import sys
sys.path.append('..')

from wrappers.basic_wrapper import DiffusionSparseWrapper
from networks.diffunet import UnetResNetBlock
from utilities.metrics import compute_SSIM
from datasets.aapmmyo import CTTools
import torch
import torch.nn as nn
import torch.nn.functional as F

def insert_interpolated_elements(view_list, k):
    if len(view_list) < 2 or k <= 0:
        return view_list.copy()

    new_list = []
    for i in range(len(view_list) - 1):
        current = view_list[i]
        next_val = view_list[i + 1]
        new_list.append(current)

        # Calculate step size
        step = (next_val - current) / (k + 1)

        # Generate and append interpolated values
        for j in range(1, k + 1):
            interpolated = current + step * j
            new_list.append(round(interpolated))

    # Add the last element from original list
    new_list.append(view_list[-1])
    return new_list

class ColdDiffusion(DiffusionSparseWrapper):
    def __init__(self,
         opt,
         **wrapper_kwargs
     ):
        super().__init__(**wrapper_kwargs)
        self.opt = opt
        self.denoise_fn = UnetResNetBlock(
            in_channels= 1,
            ch= opt.unet_dim
        )

        self.view_list = [288, 234, 180, 126, 72, 54, 36, 18]

        self.num_timesteps = len(self.view_list)

        self.cttool = CTTools()

    def generate_sparse_and_gt_data(self, mu_ct, num_views):
        if self.opt.dist:
            sparse_mu, gt_mu = self.module.generate_sparse_and_full_ct(mu_ct, num_views= num_views)
        else:
            sparse_mu, gt_mu = self.generate_sparse_and_full_ct(mu_ct, num_views= num_views)

        return sparse_mu, gt_mu

    def forward(self, x, t):
        return self.denoise_fn(x, t)

    @torch.no_grad()
    def sample(self, x, t= None):
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps - 1

        b = x.shape[0]
        x_deg = x
        t_start = t
        t_id_list = [t_start, 0]

        sequential_budget = t_start + 1
        t_id_list = insert_interpolated_elements(t_id_list, sequential_budget - 2)
        for id_id, t_id in enumerate(t_id_list):
            step = torch.full((b, ), t_id, dtype=torch.long, device=x.device)
            x0_hat = self.denoise_fn(x_deg, step)
            if id_id == 0:
                direct_recon = x0_hat
            x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views= self.view_list[t_id])

            if id_id < len(t_id_list) - 1:
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_id_list[id_id + 1]])
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                x_deg = x0_hat

        return x_deg, direct_recon
    
    @torch.no_grad()
    def iterative_sample(self, x_deg, t_start, iterative_budget, refine_budget=2):
        self.denoise_fn.eval()
        x_deg_in = x_deg

        current_t = t_start
        b = x_deg.shape[0]

        previous_direct_recon = None
        direct_recon = None

        total_budget = 0

        while (iterative_budget > 0):
            total_budget += 1
            step = torch.full((b,), current_t, dtype=torch.long, device=x_deg.device)
            x0_hat = self.denoise_fn(x_deg, step)
            if direct_recon is None:
                direct_recon = x0_hat
            if previous_direct_recon is None:
                sequential_flag = True
            elif compute_SSIM(self.cttool.window_transform(self.cttool.mu2HU(x0_hat)),
                              self.cttool.window_transform(self.cttool.mu2HU(previous_direct_recon)), data_range=1,
                              spatial_dims=2) < self.opt.time_back_ssim_threshold:
                sequential_flag = True
            else:
                sequential_flag = False
                iterative_budget = iterative_budget - 1

            if iterative_budget == 0:
                x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[refine_budget - 1])
                current_t = refine_budget - 1
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                if sequential_flag:
                    x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
                    if current_t > 0:
                        x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t - 1])
                    else:
                        x_next_deg_estimate = x0_hat
                    current_t = current_t - 1
                    x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
                    previous_direct_recon = x0_hat
                else:
                    # If we find no further improvement, we go back to start time-step
                    x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_start])
                    x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[t_start - 1])
                    current_t = t_start - 1
                    x_deg = x_deg_in - x_current_deg_estimate + x_next_deg_estimate
                    previous_direct_recon = None

        for i in range(refine_budget):
            total_budget += 1
            step = torch.full((b,), current_t, dtype=torch.long, device=x_deg.device)
            x0_hat = self.denoise_fn(x_deg, step)
            x_current_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t])
            if current_t > 0:
                x_next_deg_estimate, _ = self.generate_sparse_and_full_ct(x0_hat, num_views=self.view_list[current_t - 1])
                x_deg = x_deg - x_current_deg_estimate + x_next_deg_estimate
            else:
                x_deg = x0_hat

            current_t = current_t - 1
            
        return x_deg, direct_recon