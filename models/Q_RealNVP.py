import torch.optim as optim
import torch.nn as nn
import models.util_for_realnvp as util
from models.RealNVP import RealNVP
import numpy as np

class Q_RealNVP(nn.Module):

    def __init__(self, num_scales, num_channels, num_channels_g, num_blocks, lr_TQ,
                 max_grad_norm=10., un_normalize_x=False, no_latent=False, Q_init='identity'):
        super().__init__()
        self.model = RealNVP(num_scales=num_scales,
                             in_channels=num_channels,
                             mid_channels=num_channels_g,
                             num_blocks=num_blocks,
                             un_normalize_x=un_normalize_x,
                             no_latent=no_latent)

        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_TQ)

        util.init_model(self.model, init_method=Q_init)

    def update_Q(self, z, iter_=1):
        for _ in range(iter_):
            self.optimizer.zero_grad()
            loss = -1 * self.log_likelihood(z)
            loss.backward()
            util.clip_grad_norm(self.optimizer, self.max_grad_norm)
            self.optimizer.step()


    def log_likelihood(self, z):
        w, sldj, _ = self.model(z, preprocess=False)
        prior_ll = -0.5 * (w ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1)
        ll = prior_ll + sldj
        return ll.mean()

    def inverse(self, z):
        return self.model(z, reverse=True)[0]
