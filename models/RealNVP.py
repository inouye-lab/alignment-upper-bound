import torch
import torch.nn as nn
import torch.nn.functional as F

from models.util_for_realnvp import squeeze_2x2, checkerboard_like, WNConv2d
from enum import IntEnum
import math

def inverse_preprocess(x, logdet, noise, data_constraint=0.999998):
    # if inv_jacob:
    # Compute logdet first
    # x = x.double()

    # def better_log_sigmoid(x):
    #     if x > 0:
    #         return -torch.log1p(torch.exp(-x))
    #     else:
    #         return x - torch.log1p(torch.exp(x))

    sigma = torch.sigmoid(-torch.abs(x)).reshape(x.size(0), -1)
    logdet += torch.sum( 2*(nn.LogSigmoid()(-torch.abs(x)).reshape(x.size(0), -1) + torch.log1p(-sigma)), dim=-1 )

    # f_x = torch.sigmoid(x).view(x.size(0), -1)
    # logdet += torch.sum(torch.log(2. * (f_x * (1 - f_x)) + eps), dim=-1)
    # inverse logit
    x = torch.sigmoid(x)
    x = 2*x - 1
    x /= data_constraint
    x = (x + 1) / 2.

    # Convert to logits
    # y = (2 * x - 1) * self.data_constraint  # [-0.9, 0.9]
    # y = (y + 1) / 2  # [0.05, 0.95]
    # y = y.log() - (1. - y).log()  # logit [-inf, inf]


    # quantization?
    x = (256. * x - noise) / 255.
    # normalize from [0,1] to [-1,1]
    x = 2. * (x - 0.5)

    return x, logdet


# class Symmetric_RealNVP(nn.Module):
#     def __init__(self, fwd_realnvp, inv_realnvp):
#         super(Symmetric_RealNVP, self).__init__()
#         self.fwd_realnvp = fwd_realnvp
#         self.inv_realnvp = inv_realnvp
#
#     def forward(self, x, reverse=False):
#         if reverse == False:
#             x, fwd_logdet = self.fwd_realnvp(x, reverse=False)
#             # x, inv_logdet = self.inv_realnvp(x, reverse=True)
#             x, _ = self.inv_realnvp(x, reverse=True)
#             _, inv_logdet = self.inv_realnvp(x, reverse=False, preprocess=False)
#             inv_logdet *= -1
#         else:
#             x, inv_logdet = self.inv_realnvp(x, reverse=False, preprocess=False)
#             # x, fwd_logdet = self.fwd_realnvp(x, reverse=True)
#             x, _ = self.fwd_realnvp(x, reverse=True)
#             _, fwd_logdet = self.fwd_realnvp(x, reverse=False, preprocess=False)
#             fwd_logdet *= -1
#
#         return x, fwd_logdet + inv_logdet


class RealNVP(nn.Module):
    """RealNVP Model

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
        un_normalize_x (bool): Un-normalize inputs `x`: shift (-1, 1) to (0, 1)
            assuming we used `transforms.Normalize` with mean 0.5 and std 0.5.
        no_latent (bool): If True, assume both `x` and `z` are image distributions.
            So we should pre-process the same in both directions. E.g., True in CycleFlow.
    """
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8,
                 un_normalize_x=False, no_latent=False, data_constraint=0.999998):
        super(RealNVP, self).__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([data_constraint], dtype=torch.float64))
        # self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float64))
        self.un_normalize_x = un_normalize_x
        self.no_latent = no_latent

        # Get inner layers
        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def forward(self, x, reverse=False, preprocess=True):
        # sldj = 0
        sldj = 0 if reverse == False else None
        noise = 0 if reverse == False else None
        if (self.no_latent or not reverse) and preprocess:
            # De-quantize and convert to logits
            x, sldj, noise = self._pre_process(x)

        x, sldj = self.flows(x, sldj, reverse)

        return x, sldj, noise

    def _pre_process(self, x):
        """De-quantize and convert the input image `x` to logits.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Logits of `x`.
            ldj (torch.Tensor): Log-determinant of the Jacobian of the transform.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        if self.un_normalize_x:
            x = x * 0.5 + 0.5 # -1 1 => 0 1

        # Expect inputs in [0, 1]
        if x.min() < 0 or x.max() > 1:
            raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
                             .format(x.min(), x.max()))

        # De-quantize
        noise = torch.rand_like(x)
        x = (x * 255. + noise) / 256.

        # Convert to logits
        # lambda_preprocess = 1e-6
        # y = lambda_preprocess + (1 - 2*lambda_preprocess) * x

        y = (2 * x - 1) * self.data_constraint  # [-0.9, 0.9] 0.05 =>
        y = (y + 1) / 2.0 # [0.05, 0.95] [0.000001 0.999999]

        y = y.log() - (1. - y).log()            # logit [-inf, inf]

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
        ldj = ldj.view(ldj.size(0), -1).sum(-1)

        return y, ldj, noise


class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.

    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):

        if reverse:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=False)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=False)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=False)

                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=False)

        return x, sldj


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if self.mask_type == MaskType.CHECKERBOARD:
            norm_channels = in_channels
            out_channels = 2 * in_channels
            in_channels = 2 * in_channels + 1
        else:
            norm_channels = in_channels // 2
            out_channels = in_channels
            in_channels = in_channels
        self.st_norm = nn.BatchNorm2d(norm_channels, affine=False)
        self.st_net = STResNet(in_channels, mid_channels, out_channels,
                               num_blocks=num_blocks, kernel_size=3, padding=1)

        # Learnable scale and shift for s
        self.s_scale = nn.Parameter(torch.ones(1))
        self.s_shift = nn.Parameter(torch.zeros(1))

    def forward(self, x, sldj=None, reverse=True):
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            b = checkerboard_like(x, reverse=self.reverse_mask)
            x_b = (x * b)
            b = b.expand(x.size(0), -1, -1, -1)
            x_b = 2. * self.st_norm(x_b)
            x_b = F.relu(torch.cat((x_b, -x_b, b), dim=1))
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.s_scale * torch.tanh(s) + self.s_shift
            s = s * (1. - b)
            t = t * (1. - b)

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t

            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_norm(x_id)
            st = F.relu(torch.cat((st, -st), dim=1)) # insert mean values instead of zeros?
            st = self.st_net(st)
            s, t = st.chunk(2, dim=1)
            s = self.s_scale * torch.tanh(s) + self.s_shift

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, sldj


class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = WNConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = WNConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        x = x + skip

        return x


class STResNet(nn.Module):
    """ResNet for scale and translate factors in Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
        padding (int): Padding for convolutional layers.
    """
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks, kernel_size, padding):
        super(STResNet, self).__init__()
        self.in_conv = WNConv2d(in_channels, mid_channels, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
                                    for _ in range(num_blocks)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_conv(x)
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x