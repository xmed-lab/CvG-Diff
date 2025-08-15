import os
import tqdm
import time
import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import sys

sys.path.append('..')
from trainers.basic_trainer import BasicTrainer
from datasets.aapmmyo import AAPMMyoDataset, CTTools
from utilities.metrics import compute_measure


class ColdDiffTester(BasicTrainer):
    def __init__(self, opt, net, test_window=None, data_range=1, **kwargs):
        super().__init__()
        self.opt = opt
        self.net = net
        self.cttool = CTTools()
        if test_window is not None:
            assert isinstance(test_window, list)
            print('Test window: ', test_window)
        else:
            test_window = [(3000, 500), (500, 50), (2000, 0)] # [window width, window center]
        self.test_window = test_window
        self.save_fig = self.opt.tester_save_image
        self.data_range = data_range
        self.tables = defaultdict(dict)  # simply record metrics
        self.tables_stat = defaultdict(dict)  # record statistics
        for (width, center) in self.test_window:
            for metric_name in ['psnr', 'ssim', 'rmse']:
                self.tables[f'({width},{center})_' + metric_name] = []
                self.tables[f'({width},{center})_' + metric_name + '_direct'] = []
        self.tables['times'] = []

        self.save_dir = os.path.join(self.opt.tester_save_path, self.opt.tester_save_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Save figures to {self.save_dir}? : ', self.save_fig)
        self.saved_slice = 0
        self.seed_torch(seed=1)

    def seed_torch(self, seed=1):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def prepare_dataset(self, split= 'test'):
        opt = self.opt
        dataset_name = opt.dataset_name.lower()
        if dataset_name == 'aapm':
            self.test_dataset = AAPMMyoDataset(opt.dataset_path, mode=split, dataset_shape=opt.dataset_shape)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented, try aapm.')

        self.test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=1,)

    def save_multiple_images(self, nrows, ncols, images, titles=None, tight=False, cmaps='gray', save_path= None, font_size= 10):
        num_imgs = len(images)
        num_plots = int(nrows * ncols)
        assert num_imgs <= num_plots, f'num_imgs = {num_imgs}, nrows = {nrows}, ncols = {ncols}.'
        cmaps = [cmaps] * num_imgs if not isinstance(cmaps, (list, tuple)) else cmaps
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2), dpi=300)
        axes = axes.flatten()
        for i in range(num_imgs):
            if isinstance(images[i], torch.Tensor):
                axes[i].imshow(images[i].cpu().squeeze(), cmap=cmaps[i])
            else:
                axes[i].imshow(images[i].squeeze(), cmap=cmaps[i])
            axes[i].axis('off')
            if titles is not None:
                axes[i].set_title(titles[i], fontsize=font_size)

        if num_imgs < num_plots:
            for i in range(num_imgs, num_plots):
                axes[i].axis('off')
        if tight:
            fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    def generate_sparse_and_gt_data(self, mu_ct, num_views):
        if self.opt.dist:
            sparse_mu, gt_mu = self.net.module.generate_sparse_and_full_ct(mu_ct, num_views= num_views)
        else:
            sparse_mu, gt_mu = self.net.generate_sparse_and_full_ct(mu_ct, num_views= num_views)

        return sparse_mu, gt_mu
    
    def iterative_sample(self):
        self.prepare_dataset(self.opt.split)
        self.net = self.load_model(net=self.net, net_checkpath=self.opt.net_checkpath, output=True)

        self.net = self.net.cuda()
        self.net = self.net.eval()

        pbar = tqdm.tqdm(self.test_loader, ncols=60)
        with torch.no_grad():
            for i, data in enumerate(pbar):
                mu_ct = data.to('cuda')
                timestep_st = self.net.view_list.index(self.opt.num_views)
                sparse_mu, gt_mu = self.generate_sparse_and_gt_data(mu_ct, num_views=self.opt.num_views)
                time_stamp = time.time()
                recon_mu, direct_recon_mu = self.net.iterative_sample(sparse_mu, timestep_st, self.opt.budget_ratio, self.opt.refine_budget)
                self.tables['times'].append(time.time() - time_stamp)

                for (width, center) in self.test_window:
                    recon_hu = self.cttool.window_transform(self.cttool.mu2HU(recon_mu), width=width, center=center)
                    direct_recon_hu = self.cttool.window_transform(self.cttool.mu2HU(direct_recon_mu), width=width, center=center)
                    gt_hu = self.cttool.window_transform(self.cttool.mu2HU(gt_mu), width=width, center=center)
                    rmse, psnr, ssim = compute_measure(recon_hu, gt_hu, self.data_range)
                    rmse_direct, psnr_direct, ssim_direct = compute_measure(direct_recon_hu, gt_hu, self.data_range)
                    self.tables[f'({width},{center})_psnr'].append(psnr)
                    self.tables[f'({width},{center})_ssim'].append(ssim)
                    self.tables[f'({width},{center})_rmse'].append(rmse)
                    self.tables[f'({width},{center})_psnr_direct'].append(psnr_direct)
                    self.tables[f'({width},{center})_ssim_direct'].append(ssim_direct)
                    self.tables[f'({width},{center})_rmse_direct'].append(rmse_direct)

                    if self.save_fig:
                        self.save_png(recon_hu, 'pred', window_name=f'({width},{center})')

                self.saved_slice += 1
        self.save_csv()

    def save_png(self, value, name, window_name):
        save_dir = os.path.join(self.save_dir, window_name)
        os.makedirs(save_dir, exist_ok=True)
        saved_slice = str(self.saved_slice).rjust(3, '0')
        if name in ['gt', 'input']:
            fullname = f'{saved_slice}_{name}.png'
        else:
            fullname = f'{saved_slice}_{name}_{self.opt.network}.png'
        save_path = os.path.join(save_dir,fullname)
        save_image(value, save_path, normalize=False)

    def write_stat_table(self,):
        for (width, center) in self.test_window:
            for metric_name in ['psnr', 'ssim', 'rmse', 'psnr_direct', 'ssim_direct', 'rmse_direct']:
                table_tmp = self.tables[f'({width},{center})_' + metric_name]
                self.tables_stat[f'({width},{center})']['avg_' + metric_name] = np.mean(table_tmp)
                self.tables_stat[f'({width},{center})']['std_' + metric_name] = np.std(table_tmp)
                self.tables_stat[f'({width},{center})']['min_' + metric_name] = np.min(table_tmp)
                self.tables_stat[f'({width},{center})']['max_' + metric_name] = np.max(table_tmp)
            table_tmp = self.tables['times']
            metric_name = 'times'
            self.tables_stat[f'({width},{center})']['avg_' + metric_name] = np.mean(table_tmp)
            self.tables_stat[f'({width},{center})']['std_' + metric_name] = np.std(table_tmp)
            self.tables_stat[f'({width},{center})']['min_' + metric_name] = np.min(table_tmp)
            self.tables_stat[f'({width},{center})']['max_' + metric_name] = np.max(table_tmp)
            print(f"Averaged PSNR under window {(width, center)}: {self.tables_stat[f'({width},{center})']['avg_psnr']}")
            print(f"Averaged SSIM under window {(width, center)}: {self.tables_stat[f'({width},{center})']['avg_ssim']}")
            print(f"Averaged RMSE under window {(width, center)}: {self.tables_stat[f'({width},{center})']['avg_rmse']}")

    def save_csv(self,):
        self.write_stat_table()
        df = pd.DataFrame(self.tables)
        csv_path = os.path.join(self.save_dir, self.opt.network + str(self.opt.num_views) +'_all.csv')
        df.to_csv(csv_path)
        print('Table written in: ', csv_path)

        df_stat = pd.DataFrame(self.tables_stat)
        csv_stat_path = os.path.join(self.save_dir, self.opt.network + str(self.opt.num_views) +'_stat.csv')
        df_stat.to_csv(csv_stat_path)
        print('Table (stat) written in: ', csv_stat_path)
        print(df_stat)

