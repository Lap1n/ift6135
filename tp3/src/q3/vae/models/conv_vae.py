import torch.nn as nn
from torch.autograd import Variable

from q3.models import Generator_dcgan0


class Encoder(nn.Module):
    def __init__(self, input_channels, num_filters, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2,
                      padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Conv2d(num_filters, 2 * num_filters, kernel_size=4, stride=2,
                      padding=1),
            nn.BatchNorm2d(2 * num_filters),
            nn.ReLU(True),
            nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(4 * num_filters),
            nn.ReLU(True),
            nn.Conv2d(4 * num_filters, hidden_size, kernel_size=4),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.seq(x)
        return out


class ConvVAE(nn.Module):
    def __init__(self, width=32, height=32, num_channels=3, hidden_size=500,
                 z_dim=20, num_filters=64):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.width = width
        self.height = height
        self.nChannels = num_channels
        self.encoder = Encoder(num_channels, num_filters, hidden_size)
        self.decoder = Generator_dcgan0(latent_dim=z_dim)
        self.linear_mu = nn.Linear(hidden_size, z_dim)
        self.linear_log_var = nn.Linear(hidden_size, z_dim)

    def reparameterization_trick(self, mu, logvar):
        std = (logvar * 0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps * std + mu

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.hidden_size)
        mu = self.linear_mu(h)
        logvar = self.linear_log_var(h)
        z = self.reparameterization_trick(mu, logvar)
        return self.decoder(z), mu, logvar
