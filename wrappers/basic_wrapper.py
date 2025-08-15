import torch
import numpy as np
import torch.nn as nn
from torch_radon import Radon, RadonFanbeam
import torch.nn.functional as F

"""
The wrappers are used to provide methods for preparing sparse-view input data.
"""

class DiffusionSparseWrapper(nn.Module):
    def __init__(self, img_size=256, num_full_views=720, source_distance=1075, det_count=672):
        super().__init__()
        self.num_full_views = num_full_views
        self.source_distance = source_distance
        self.det_count = det_count
        self.img_size = img_size

    # ------------ basic radon function ----------------
    # avoid possible cuda error, put radon func in the module
    def radon(self, sinogram, num_views=None, angle_bias=0):
        '''sinogram to ct image'''
        angles = self.get_angles(num_views, angle_bias)
        radon_tool = RadonFanbeam(self.img_size, angles, self.source_distance, det_count=self.det_count, )
        filter_sin = radon_tool.filter_sinogram(sinogram, "ram-lak")
        back_proj = radon_tool.backprojection(filter_sin)
        return back_proj

    def image_radon(self, ct_image, num_views=None, angle_bias=0):
        '''ct image to sinogram'''
        angles = self.get_angles(num_views, angle_bias)
        radon_tool = RadonFanbeam(self.img_size, angles, self.source_distance, det_count=self.det_count, )
        sinogram = radon_tool.forward(ct_image)
        return sinogram

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward in wrapper should be implemented.')

    # ------------ basic sparse-view CT data generation ----------------
    def generate_sparse_and_full_ct(self, mu_ct, num_views, angle_bias=0):
        full_sinogram = self.generate_sinogram(mu_ct, self.num_full_views, angle_bias)
        full_mu = self.radon(full_sinogram, )
        sparse_sinogram = self.generate_sinogram(full_mu, num_views, angle_bias)
        sparse_mu = self.radon(sparse_sinogram, num_views, angle_bias)

        return sparse_mu, full_mu

    def generate_sparse_and_full_sinogram(self, mu_ct, num_views, angle_bias=0):
        full_sinogram = self.image_radon(mu_ct)
        sparse_sinogram = self.image_radon(mu_ct, num_views, angle_bias)
        return sparse_sinogram, full_sinogram

    def generate_sinogram(self, mu_ct, num_views, angle_bias= 0):
        sinogram = self.image_radon(mu_ct, num_views, angle_bias)
        return sinogram

    def get_angles(self, num_views=None, angle_bias=0, is_bias_radian=True):
        num_views = self.num_full_views if num_views is None else num_views  # specified number of views
        angles = np.linspace(0, np.pi * 2, num_views,
                             endpoint=False)  # select views according to the specified number of views
        angle_bias = angle_bias / 360 * 2 * np.pi if not is_bias_radian else angle_bias

        return angles + angle_bias
