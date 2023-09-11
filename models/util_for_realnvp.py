import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils as utils
from torch.optim import lr_scheduler as torch_scheduler
import numpy as np

def squeeze_2x2(x, reverse=False, alt_order=False):
    """For each spatial position, a sub-volume of shape `1x1x(N^2 * C)`,
    reshape into a sub-volume of shape `NxNxC`, where `N = block_size`.

    Adapted from:
        https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_utils.py

    See Also:
        - TensorFlow nn.depth_to_space: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        reverse (bool): Whether to do a reverse squeeze (unsqueeze).
        alt_order (bool): Whether to use alternate ordering.
    """
    block_size = 2
    # x=x_in.clone()
    if alt_order:
        n, c, h, w = x.size()

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels must be divisible by 4, got {}.'.format(c))
            c //= 4
        else:
            if h % 2 != 0:
                raise ValueError('Height must be divisible by 2, got {}.'.format(h))
            if w % 2 != 0:
                raise ValueError('Width must be divisible by 4, got {}.'.format(w))
        # Defines permutation of input channels (shape is (4, 1, 2, 2)).
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)

        perm_weight = torch.zeros((4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(c):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)] # 0 4 8 12
                                        + [c_idx * 4 + 1 for c_idx in range(c)] # 1 5 9 13
                                        + [c_idx * 4 + 2 for c_idx in range(c)] # 2 6 10 14
                                        + [c_idx * 4 + 3 for c_idx in range(c)]) # 3 7 11 15
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if reverse:
            # shuffle_tensor = torch.tensor([[idx * c + c_idx for idx in range(4)] for c_idx in range(c)])
            # shuffle_channels = shuffle_tensor.view(-1)
            # perm_weight = perm_weight[shuffle_channels, :, :, :]
            x = F.conv_transpose2d(x, perm_weight, stride=2)

        else:
            # shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
            #                                 + [c_idx * 4 + 1 for c_idx in range(c)]
            #                                 + [c_idx * 4 + 2 for c_idx in range(c)]
            #                                 + [c_idx * 4 + 3 for c_idx in range(c)])
            # perm_weight = perm_weight[shuffle_channels, :, :, :]
            x = x.contiguous()
            x = F.conv2d(x, perm_weight, stride=2)

    else:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels {} is not divisible by 4'.format(c))
            x = x.view(b, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(b, 2 * h, 2 * w, c // 4)
        else:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, c)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, h // 2, w // 2, c * 4)

        x = x.permute(0, 3, 1, 2)

    return x


def checkerboard_like(x, reverse=False):
    """Get a checkerboard mask for `x`, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.

    Args:
        x (torch.Tensor): Tensor that will be masked with `x`.
        reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.

    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
    """
    height, width = x.size(2), x.size(3)
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=x.dtype, device=x.device, requires_grad=False)

    if reverse:
        mask = 1. - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, 1, height, width)

    return mask

class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)

        return x


def init_model(model, init_method='normal'):
    """Initialize model parameters.

    Args:
        model: Model to initialize.
        init_method: Name of initialization method: 'normal' or 'xavier'.
    """
    # Initialize model parameters
    if init_method == 'normal':
        model.apply(_normal_init)
    elif init_method == 'xavier':
        model.apply(_xavier_init)
    elif init_method == 'identity':
        model.apply(_identity_init)
    else:
        raise NotImplementedError('Invalid weights initializer: {}'.format(init_method))


def _identity_init(model):
    """Apply normal initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and model.weight is not None:
        if class_name.find('Conv') != -1:
            nn.init.zeros_(model.weight.data)
        elif class_name.find('Linear') != -1:
            nn.init.zeros_(model.weight.data)
        elif class_name.find('BatchNorm') != -1:
            nn.init.zeros_(model.weight.data)
            nn.init.zeros_(model.bias.data)


def _normal_init(model):
    """Apply normal initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and model.weight is not None:
        if class_name.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find('Linear') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)


def _xavier_init(model):
    """Apply Xavier initializer to all model weights."""
    class_name = model.__class__.__name__
    if hasattr(model, 'weight') and model.weight is not None:
        if class_name.find('Conv') != -1:
            nn.init.xavier_normal(model.weight.data, gain=0.02)
        elif class_name.find('Linear') != -1:
            nn.init.xavier_normal(model.weight.data, gain=0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0.0)


def get_param_groups(net, weight_decay, norm_suffix='weight_g', verbose=False):
    """Get two parameter groups from `net`: One named "normalized" which will
    override the optimizer with `weight_decay`, and one named "unnormalized"
    which will inherit all hyperparameters from the optimizer.

    Args:
        net (torch.nn.Module): Network to get parameters from
        weight_decay (float): Weight decay to apply to normalized weights.
        norm_suffix (str): Suffix to select weights that should be normalized.
            For WeightNorm, using 'weight_g' normalizes the scale variables.
        verbose (bool): Print out number of normalized and unnormalized parameters.
    """
    norm_params = []
    unnorm_params = []
    for n, p in net.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)

    param_groups = [{'name': 'normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': 'unnormalized', 'params': unnorm_params}]

    if verbose:
        print('{} normalized parameters'.format(len(norm_params)))
        print('{} unnormalized parameters'.format(len(unnorm_params)))

    return param_groups


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.
    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)


def get_lr_scheduler(optimizer, lambda_lr):
    """Get learning rate scheduler."""
    # def get_lr_multiplier(epoch):
    #     init_epoch = 1
    #     return lambda_lr**max(0, epoch + init_epoch - warmup_epochs)
    # scheduler = torch_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    scheduler = torch_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    return scheduler



class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self):
        super(RealNVPLoss, self).__init__()

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        # print(prior_ll.size())
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1)
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll


class RealNVPLoss_test(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self):
        super(RealNVPLoss_test, self).__init__()

    def forward(self, z):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        # print(prior_ll.size())
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1)
        ll = prior_ll
        nll = ll.mean()

        return nll
