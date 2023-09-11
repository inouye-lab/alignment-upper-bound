from itertools import chain
import argparse
import torch
import torch.optim as optim

from models.RealNVP import RealNVP, inverse_preprocess
from models.util_for_realnvp import init_model, get_param_groups, clip_grad_norm
from data_loaders.MNIST import MNIST
from torch.optim import lr_scheduler as torch_scheduler
from models.Q_RealNVP import Q_RealNVP

from utils import save_loss, normalize_
from torchvision import transforms

import random
import os
import time
import datetime
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

def str2bool(v):
    return v.lower() in ('true')

def _transportation_cost(X_list, Z_list, weight=None):
    loss = 0
    for X, Z in zip(X_list, Z_list):
        diff = torch.abs(X - Z)
        if weight is None:  # default is uniform weight
            weight = 1 / len(X_list)
        loss += weight * torch.mean(torch.sum(diff * diff, dim=1))
    return loss

def main(config):
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    data_dir = config.data_dir
    result_dir = config.result_dir
    model_save_dir = config.model_save_dir

    num_epoch = config.num_epoch
    batch_size = config.batch_size
    num_workers = config.num_workers
    num_scales = config.num_scales
    num_blocks = config.num_blocks
    num_channels = config.num_channels
    num_channels_g = config.num_channels_g

    Q_num_scales = config.Q_num_scales
    Q_num_blocks = config.Q_num_blocks
    Q_num_channels_g = config.Q_num_channels_g

    lambda_TC = config.lambda_TC
    multi_gpu = config.multi_gpu
    T_init_method = config.T_init_method
    load_T = config.load_T
    load_Q = config.load_Q
    start_epoch = config.start_epoch
    target_numbers = [int(number) for number in config.target_numbers]

    Q_init_method = config.Q_init_method
    Q_lr = config.Q_lr

    gpu_id = config.gpu_id
    num_iter_for_Q = config.num_iter_for_Q

    lr = config.lr
    setting = config.setting

    train_dir = os.path.join(result_dir, 'experiment_results', f'train_results_{setting}')
    val_dir = os.path.join(result_dir, 'experiment_results', f'val_results_{setting}')


    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    rescaling = lambda x: (x - .5) * 2.

    kwargs = {'num_workers': num_workers, 'pin_memory': False, 'drop_last': True}
    ds_transforms = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), rescaling])

    train_loaders = [torch.utils.data.DataLoader(MNIST(data_dir=data_dir, which_number=trg_num, mode='train', transform=ds_transforms), batch_size=batch_size,
                                               shuffle=True, **kwargs) for trg_num in target_numbers]

    val_loaders = [torch.utils.data.DataLoader(MNIST(data_dir=data_dir, which_number=trg_num, mode='val', transform=ds_transforms), batch_size=batch_size,
                                               shuffle=False, **kwargs) for trg_num in target_numbers]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    T_arr = [RealNVP(num_scales=num_scales,
                     in_channels=num_channels,
                     mid_channels=num_channels_g,
                     num_blocks=num_blocks,
                     un_normalize_x=True,
                     no_latent=False) for _ in range(len(target_numbers))]

    Q = Q_RealNVP(Q_num_scales,
                  num_channels,
                  Q_num_channels_g,
                  Q_num_blocks,
                  Q_lr,
                  max_grad_norm=10.,
                  un_normalize_x=False,
                  no_latent=False,
                  Q_init=Q_init_method)

    if load_Q:
        pretrained_dict = torch.load(f'./trained_models/Q_MNIST.pth')
        Q.model.load_state_dict(pretrained_dict)

    if torch.cuda.is_available():
        if multi_gpu:
            Q = torch.nn.DataParallel(Q).cuda()
        else:
            Q = Q.to(device)

    def get_lr_multiplier(epoch):
        if load_T:
            return 1.0 - epoch / float(200)
        else:
            return 1.0 - max(0, epoch - 100) / float(100)

    param_groups = []
    for i, T in enumerate(T_arr):
        if load_T:
            pretrained_dict = torch.load(f'./trained_models/T{i}_alignflow.pth')
            T.load_state_dict(pretrained_dict)

        else:
            init_model(T, init_method=T_init_method)

        if torch.cuda.is_available():
            if multi_gpu:
                T = torch.nn.DataParallel(T).cuda()

            else:
                T = T.to(device)

        param_groups.append(get_param_groups(T, 5e-5, norm_suffix='weight_g'))



    optimizers = []
    for param_group in param_groups:
        optimizers.append(optim.Adam(chain(param_group), lr=lr, betas=(0.5, 0.999)))

    lr_schedulers = []
    for optimizer in optimizers:
        lr_schedulers.append(torch_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier))

    train_losses = []
    train_nlls = []
    train_logdet_losses = []
    train_tc_losses = []
    val_losses = []
    val_tc_losses = []

    min_val_loss = 10000
    start_time = time.time()
    for trg_num, train_loader in zip(target_numbers, train_loaders):
        print(f'# mini batches in dataloader for {trg_num}', len(train_loader))
    print_at = 0

    for epoch in range(start_epoch, start_epoch+num_epoch+1):

        for train_loader in train_loaders:
            train_loader.dataset.shuffle_dataset()

        train_loss = 0
        train_nll = 0
        train_logdet = 0
        train_tc_loss = 0

        for i, x_arr in enumerate(zip(*train_loaders)):
            for param in Q.parameters():
                param.requires_grad = True

            z_mix = []
            for j, x in enumerate(x_arr):
                assert (len(x_arr)==len(T_arr)), "Check the length of Xs, Ts and optimizers."
                x = x.to(device).double()

                T = T_arr[j]
                T.eval()

                for param in T.parameters():
                    param.requires_grad = False

                if i == 0:
                    print(f'epoch {epoch} | learning_rate for {target_numbers[j]}:', optimizer.param_groups[0]['lr'])

                z, _, _ = T(x) # z1 -3 3
                z_mix.append(z.detach())

            z_mix = torch.stack(z_mix, dim=0).detach()  # K, B, C, H, W
            z_chunks = torch.split(z_mix, batch_size // len(target_numbers), dim=1)  # tuple(K, B//K, C, H, W)
            for k, z_mix in enumerate(z_chunks):
                z_mix = z_mix.reshape(z_mix.size(0) * z_mix.size(1), 1, 32, 32)
                Q.update_Q(z_mix, iter_=num_iter_for_Q)  # B, C,

            for param in Q.parameters():
                param.requires_grad = False

            for j, x in enumerate(x_arr):
                assert (len(x_arr) == len(T_arr)), "Check the length of Xs, Ts and optimizers."
                optimizer = optimizers[j]
                optimizer.zero_grad()
                x = x.to(device).double()

                T = T_arr[j]

                T.train()

                for param in T.parameters():
                    param.requires_grad = True

                z, logdet, _ = T(x)  # z1 -3 3

                loss = (1/float(len(x_arr))) * (torch.mean(-Q.log_likelihood(z)) - torch.mean(logdet))
                tc_loss = (1/float(len(x_arr))) * _transportation_cost([x], [z])
                tc_loss *= lambda_TC

                train_loss += loss.item() + tc_loss.item()

                (loss + tc_loss).backward()

                train_nll += (1/float(len(x_arr))) * torch.mean(-Q.log_likelihood(z.detach())).detach().item()
                train_logdet += (1/float(len(x_arr))) * (- torch.mean(logdet.detach())).item()

                train_tc_loss += tc_loss.detach().item()

                clip_grad_norm(optimizer, 10.)
                optimizer.step()

                for param in T.parameters():
                    param.requires_grad = False

            if i == print_at:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                print(f"Elapsed [{et}], epoch: {epoch}, idx_steps: {i+1:}, "
                      f"nll: {train_nll / (i+1)}, logdet_loss: {train_logdet / (i+1)}, tc_loss: {train_tc_loss / (i+1):.5f}")

                z_arr = []
                noise_arr = []
                ################################
                translated_results = []
                with torch.no_grad():
                    [T.eval() for T in T_arr]
                    for x, T_forward in zip(x_arr, T_arr):
                        x = x.to(device).double()
                        translated_results_for_specific_class = []
                        z, _, noise = T_forward(x)
                        z_arr.append(z)
                        noise_arr.append(noise)
                        for T_inv in T_arr:
                            x_hat = T_inv(z_arr[-1], reverse=True)[0]
                            translated_results_for_specific_class.append(inverse_preprocess(x_hat, 0, noise)[0])

                        translated_results.append(torch.cat(translated_results_for_specific_class, dim=3))

                    for k in range(len(T_arr)):
                        x = x_arr[k]
                        z = z_arr[k].cpu()
                        x_invs = translated_results[k].cpu()
                        images = torch.cat([x, normalize_(z), x_invs], dim=3)

                        tensor_image = torch.cat([
                            images[i] for i in range(images.size(0))
                        ], dim=1)

                        plt.axis('off')

                        plt.imsave(os.path.join(train_dir, f'train_{epoch}_{target_numbers[k]}_{i}.png'),
                                    tensor_image.permute(1, 2, 0)[:,:,0], cmap='gray', vmin=-1, vmax=1)

        train_losses += [train_loss / (i + 1)]
        train_tc_losses += [train_tc_loss / (i + 1)]
        train_nlls += [train_nll / (i + 1)]
        train_logdet_losses += [train_logdet / (i + 1)]

        print(f'TRAIN--- Overall_loss: {train_loss / (i + 1)}, NLL (w/o logdet): {train_nll / (i + 1)}'
                        f', Logdet: {train_logdet / (i + 1)}, TC_loss: {train_tc_loss / (i + 1)}')

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

        val_loss = 0
        val_tc_loss = 0
        with torch.no_grad():

            for i, x_arr in enumerate(zip(*val_loaders)):

                x_mix = []
                z_mix = []
                z_arr = []
                noise_arr = []
                logdets = []
                translated_results = []
                for j, x in enumerate(x_arr):
                    x = x.to(device).double()
                    x_mix.append(x)

                    T = T_arr[j]
                    T.eval()

                    z, logdet, noise = T(x)  # z1 -3 3
                    z_mix.append(z)
                    logdets.append(logdet)
                    z_arr.append(z)
                    noise_arr.append(noise)

                x_mix = torch.cat(x_mix, dim=0)
                z_mix = torch.cat(z_mix, dim=0)
                loss = torch.mean(-Q.log_likelihood(z_mix)) - torch.mean(torch.cat(logdets, dim=0))

                tc_loss = _transportation_cost([x_mix], [z_mix])
                tc_loss *= lambda_TC

                val_loss += loss.item() + tc_loss.item()
                val_tc_loss += tc_loss.item()
                if i == 0:
                    for z in z_arr:
                        translated_results_for_specific_class = []
                        for T_inv in T_arr:
                            x_hat = T_inv(z, reverse=True)[0]
                            translated_results_for_specific_class.append(inverse_preprocess(x_hat, 0, noise)[0])
                        translated_results.append(torch.cat(translated_results_for_specific_class, dim=3))

                    for k in range(len(T_arr)):
                        x = x_arr[k]
                        z = z_arr[k].cpu()
                        noise = noise_arr[k].cpu()
                        x_invs = translated_results[k].cpu()
                        images = torch.cat([x, normalize_(z), x_invs], dim=3)

                        tensor_image = torch.cat([
                            images[i] for i in range(images.size(0))
                        ], dim=1)

                        plt.axis('off')

                        plt.imsave(os.path.join(val_dir, f'val_{epoch}_{target_numbers[k]}_{i}.png'),
                                   tensor_image.permute(1, 2, 0)[:, :, 0], cmap='gray', vmin=-1, vmax=1)

            print(f'VAL --- Overall_loss: {val_loss / i}, TC_loss: {val_tc_loss / i}')
            val_losses += [val_loss / i]
            val_tc_losses += [val_tc_loss / i]

            if val_losses[-1] <= min_val_loss:
                min_val_loss = val_losses[-1]
                for j, T in enumerate(T_arr):
                    torch.save(T.state_dict(), os.path.join(model_save_dir, f'best_{epoch}_T{target_numbers[j]}_{setting}.pth'))
                torch.save(Q.state_dict(), os.path.join(model_save_dir, f'best_{epoch}_Q_{setting}.pth'))

        if epoch == 0 and not load_T:
            train_nlls.pop(0)
            train_logdet_losses.pop(0)
            train_losses.pop(0)
            train_tc_losses.pop(0)

        save_loss(train_nlls, f'train_neg_ll', train_dir)
        save_loss(train_logdet_losses, f'train_neg_logdet', train_dir)
        save_loss(train_losses, f'train_overall', train_dir)
        save_loss(train_tc_losses, f'train_TC_loss', train_dir)

        save_loss(val_losses, f'val_overall', val_dir)
        save_loss(val_tc_losses, f'val_TC_loss', val_dir)

        for j, T in enumerate(T_arr):
            torch.save(T.state_dict(), os.path.join(model_save_dir, f'latest_T{target_numbers[j]}_{setting}.pth'))
        torch.save(Q.state_dict(), os.path.join(model_save_dir, f'latest_Q_{setting}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num_scales', type=int, default=4, help='') # 2
    parser.add_argument('--num_blocks', type=int, default=4, help='') # 2
    parser.add_argument('--num_channels', type=int, default=1, help='') #
    parser.add_argument('--num_channels_g', type=int, default=16, help='')

    parser.add_argument('--Q_num_scales', type=int, default=4, help='')
    parser.add_argument('--Q_num_blocks', type=int, default=4, help='')
    parser.add_argument('--Q_num_channels_g', type=int, default=16, help='')
    parser.add_argument('--Q_init_method', type=str, default='identity') # normal, identity, xavier
    parser.add_argument('--Q_lr', type=float, default=2e-4)
    parser.add_argument('--num_iter_for_Q', type=int, default=1)

    # Training configuration.
    parser.add_argument('--lr', type=float, default=2e-5, help='')
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lambda_TC', type=float, default=1., help='') # 100 10 1
    parser.add_argument('--target_numbers', type=str, required=True)

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--multi_gpu', type=str2bool, default=False)
    parser.add_argument('--setting', type=str, required=True)
    parser.add_argument('--T_init_method', type=str, default='normal')  # normal, xavier, identity

    parser.add_argument('--load_T', type=str2bool, default=True)
    parser.add_argument('--load_Q', type=str2bool, default=True)

    parser.add_argument('--start_epoch', type=int, default=0)

    # Directories.
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_save_dir', type=str, default='./trained_models')
    parser.add_argument('--result_dir', type=str, default='./results')


    config = parser.parse_args()

    print(config)
    main(config)