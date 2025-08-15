import os
import sys
import random
import argparse
import numpy as np
import torch

from networks.colddiff import ColdDiffusion
from trainers.colddiff_trainer import ColdDiffTrainer
from trainers.colddiff_tester import ColdDiffTester


def get_parser():
    parser = argparse.ArgumentParser(description='Sparse CT Main')
    # logging interval by iteration
    parser.add_argument('--log_interval', type=int, default=400, help='logging interval by iteration')
    parser.add_argument('--val_interval', type=int, default=3, help='validation interval by epoch')
    # tensorboard config
    parser.add_argument('--checkpoint_root', type=str, default='', help='where to save the checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='test', help='detail folder of checkpoint')
    # wandb config
    parser.add_argument('--use_tqdm', action='store_true', default=False, help='whether to use tqdm')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='whether to use wandb')
    parser.add_argument('--wandb_project', type=str, default='CvG-Diff')
    parser.add_argument('--wandb_entity', type=str, default='herlocked')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--wandb_root', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')
    # DDP
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for torch distributed training')
    parser.add_argument('--dist', action='store_true', default=False, help='whether to use distributed training')
    # data_path
    parser.add_argument('--dataset_path', type=str, default='', help='dataset path')
    parser.add_argument('--dataset_name', default='aapm', type=str,
                        help='which dataset, size640,size320,deepleision.etc.')
    parser.add_argument('--dataset_shape', type=int, default=512, help='modify shape in dataset')
    parser.add_argument('--num_train', default=5410, type=int, help='number of training examples')
    parser.add_argument('--num_val', default=526, type=int, help='number of validation examples')
    parser.add_argument('--split', default='test', type=str, help='train/val/test')
    # dataloader
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--shuffle', default=True, type=bool, help='dataloader shuffle, False if test and val')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers, 4 is a good choice')
    parser.add_argument('--drop_last', default=False, type=bool, help='dataloader droplast')
    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str, help='name of the optimizer')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta2')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('--save_epochs', default=10, type=int)
    # scheduler
    parser.add_argument('--scheduler', default='step', type=str, help='name of the scheduler')
    parser.add_argument('--step_size', default=25, type=int, help='step size for StepLR')
    parser.add_argument('--milestones', nargs='+', type=int, help='milestones for MultiStepLR')
    parser.add_argument('--step_gamma', default=0.8, type=float, help='learning rate reduction factor')
    parser.add_argument('--poly_iters', default=10, type=int,
                        help='the number of steps that the scheduler decays the learning rate')
    parser.add_argument('--poly_power', default=2, type=float, help='the power of the polynomial')

    # checkpath && resume training
    parser.add_argument('--resume', default=False, action='store_true',
                        help='resume network training or not, load network param')
    parser.add_argument('--resume_opt', default=False, action='store_true',
                        help='resume optimizer or not, load opt param')
    parser.add_argument('--net_checkpath', default='', type=str, help='network checkpoint path')
    parser.add_argument('--opt_checkpath', default='', type=str, help='optimizer checkpath')

    # network hyper args
    parser.add_argument('--trainer_mode', default='train', type=str, help='train or test')
    parser.add_argument('--loss', default='l2', type=str, help='loss type')
    parser.add_argument('--network', default='', type=str, help='networkname')

    # tester args
    # TBD here
    parser.add_argument('--tester_save_name', default='default_save', type=str, help='name of test')
    parser.add_argument('--tester_save_image', default=False, action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--tester_save_path', default='', type=str, help='path for saving tester result')
    # sparse ct args
    parser.add_argument('--num_views', default=18, type=int, help='common setting: 18/36/72/144 out of 720')
    parser.add_argument('--num_full_views', default=720, type=int, help='720 for fanbeam 2D')

    # network args
    parser.add_argument('--unet_dim', type=int, default=128, help='dimension of unet')
    parser.add_argument('--update_ema_iter', default=10, type=int)
    parser.add_argument('--start_ema_iter', default=2000, type=int)
    parser.add_argument('--ema_decay', default=0.995, type=float)

    parser.add_argument('--budget_ratio', type= int, default=2)
    parser.add_argument('--refine_budget', type=int, default=2)
    parser.add_argument('--time_back_ssim_threshold', type=float, default=0.98)
    
    return parser


def sparse_main(opt):
    net_name = opt.network
    print('Network name: ', net_name)
    wrapper_kwargs = {
        'num_full_views': opt.num_full_views,
        'img_size': opt.dataset_shape}

    net = ColdDiffusion(opt, **wrapper_kwargs)
    if opt.trainer_mode == 'train':
        trainer = ColdDiffTrainer(opt=opt, net=net, loss_type=opt.loss)
        trainer.fit()
    elif opt.trainer_mode == 'test' or opt.trainer_mode == 'val':
        tester = ColdDiffTester(opt=opt, net=net, test_window=None)
        tester.iterative_sample()
    else:
        raise ValueError('opt trainer mode error: must be train, val, or test, not {}'.format(opt.trainer_mode))

    print('done')


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    seed = 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sparse_main(opt)


