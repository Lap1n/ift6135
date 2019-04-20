import torch

import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

from q3.models import Generator_dcgan0
from q3.vae.vae_utils import UnNormalize


# def buildDecoderNetwork(hidden_size, nFilters, output_channels):
#     net = []
#     net.append(nn.ConvTranspose2d(hidden_size, 4 * nFilters, kernel_size=4))
#     net.append(nn.BatchNorm2d(4 * nFilters))
#     net.append(nn.ReLU(True))
#
#     net.append(
#         nn.ConvTranspose2d(4 * nFilters, 2 * nFilters, kernel_size=4, stride=2,
#                            padding=1))
#     net.append(nn.BatchNorm2d(2 * nFilters))
#     net.append(nn.ReLU(True))
#
#     net.append(
#         nn.ConvTranspose2d(2 * nFilters, nFilters, kernel_size=4, stride=2,
#                            padding=1))
#     net.append(nn.BatchNorm2d(nFilters))
#     net.append(nn.ReLU(True))
#
#     net.append(
#         nn.ConvTranspose2d(nFilters, output_channels, kernel_size=4, stride=2,
#                            padding=1))
#     return nn.Sequential(*net)


def buildEncoderNetwork(input_channels, nFilters, hidden_size):
    net = []
    net.append(nn.Conv2d(input_channels, nFilters, kernel_size=4, stride=2,
                         padding=1))
    net.append(nn.BatchNorm2d(nFilters))
    net.append(nn.ReLU(True))

    net.append(
        nn.Conv2d(nFilters, 2 * nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(2 * nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(2 * nFilters, 4 * nFilters, kernel_size=4, stride=2,
                         padding=1))
    net.append(nn.BatchNorm2d(4 * nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(4 * nFilters, hidden_size, kernel_size=4))
    net.append(nn.BatchNorm2d(hidden_size))
    net.append(nn.ReLU(True))

    return nn.Sequential(*net)


class ConvVAE(nn.Module):
    """
    https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/autoencoder/convVAE.py
    """

    def __init__(self, width=32, height=32, nChannels=3, hidden_size=500,
                 z_dim=20, nFilters=64):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.width = width
        self.height = height
        self.nChannels = nChannels
        self.encoder = buildEncoderNetwork(nChannels, nFilters, hidden_size)
        # self.decoder = buildDecoderNetwork(hidden_size, nFilters, nChannels)
        self.decoder = Generator_dcgan0(latent_dim=z_dim)
        self._enc_mu = nn.Linear(hidden_size, z_dim)
        self._enc_log_sigma = nn.Linear(hidden_size, z_dim)
        self._dec = nn.Linear(z_dim, hidden_size)
        self._dec_bn = nn.BatchNorm1d(hidden_size)
        self._dec_relu = nn.ReLU(True)
        self._dec_act = nn.Tanh()
        self.inverse_transform = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.hidden_size)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    # def decode(self, z):
    #     h = self._dec_relu(self._dec_bn(self._dec(z)))
    #     h = h.view(-1, self.hidden_size, 1, 1)
    #     x = self.decoder(h)
    #     x = x.view(-1, self.nChannels, self.height, self.width)
    #     if self._dec_act is not None:
    #         x = self._dec_act(x)
    #     return x

    def sample_from_random_z(self, num_sample):
        self.eval()
        with torch.no_grad():
            z = Tensor(num_sample, self.z_dim).normal_().cuda()
            z = self.decoder(z)
            return self.inverse_transform(z)
