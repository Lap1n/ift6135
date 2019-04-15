import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("../../../")
import tp3.src.given_code.classify_svhn as classify_svhn
# import

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        # deconv formula : stride * (input -1) + kernel_size  - 2 * padding

        class View(nn.Module):
            def __init__(self):
                super(View, self).__init__()

            def forward(self, x):
                return x.view(-1, 512, 3, 3)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512 * 3 * 3),  # -1xlatent_dim --> -1x512x3x3
            View(),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            # -1x512x3x3 --> -1x512x5x5
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            # -1x512x5x5 --> -1x256x9x9
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            # -1x256x9x9 --> -1x128x17x17
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=2),
            # -1x128x9x9 --> -1x3x32x32
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        class View(nn.Module):
            def __init__(self):
                super(View, self).__init__()

            def forward(self, x):
                return x.view(-1, 512 * 3 * 3)

        # conv formula : (input + 2*padding - kernel_size)/stride + 1
        self.model = nn.Sequential(
            # -1x3x32x32 --> -1x64x28x28
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # -1x64x28x28 --> -1x128x24x24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # -1x128x24x24--> -1x256x20x20
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # -1x256x20x20 --> -1x512x16x16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # -1x512x16x16 --> -1x512x3x3
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=4, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            View(),
            nn.Linear(512 * 3 * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def get_dims(n):
    a = int(np.sqrt(n))
    while a > 1:
        if n - int(n / a) * a == 0:
            return a, int(n / a)
        a -= 1
    return 1, n


def view_samples(samples):
    n = samples.size()[0]
    nrows, ncols = get_dims(n)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharey=True, sharex=True)
    imgs = [samples[i, :, :, :].detach().cpu().numpy().transpose((1, 2, 0)) for i in range(n)]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        ax.imshow(img, aspect='equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


cuda = True if torch.cuda.is_available() else False

generator = Generator().to(get_device())
discriminator = Discriminator().to(get_device())

optimizer_G = optim.Adam(generator.parameters())
optimizer_D = optim.Adam(discriminator.parameters())

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# small modifi from q1
def compute_gradient_penality(D, x_real, x_fake, lambda_grad_penality):
    a = torch.rand(x_real.size()[0], 1, 1, 1).to(get_device())
    a.expand_as(x_real)
    a.requires_grad = True
    z = a * x_real + (1 - a) * x_fake
    D_z = D(z)
    grads = autograd.grad(outputs=D_z,
                          inputs=z,
                          grad_outputs=torch.ones(D_z.size()).to(get_device()),
                          create_graph=True,
                          retain_graph=True)[0]
    gradient_penalty = lambda_grad_penality * ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def get_wd(D, x_real, x_fake, lambda_grad_penality=10):
    return D(x_real).mean() - D(x_fake).mean() - compute_gradient_penality(D, x_real, x_fake, lambda_grad_penality)


n_epoch = 1
latent_dim = 100
batch_size = 32
lambda_grad_penality = 10
n_critic = 10
view_samples_interval = 200

train, valid, test = classify_svhn.get_data_loader("svhn", batch_size)

for epoch in range(n_epoch):
    for i, (x_real, y_real) in enumerate(train):
        if cuda:
            x_real = x_real.to(get_device())
        optimizer_D.zero_grad()
        z = Tensor(batch_size, latent_dim).normal_()
        x_fake = generator(z)

        wd = get_wd(discriminator, x_real, x_fake, lambda_grad_penality)
        wd.backward(Tensor([-1]))
        optimizer_D.step()

        # update the generator avec discriminator has trained for "n_critic" steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            x_fake = generator(z)
            generator_loss = - torch.mean(discriminator(x_fake))
            generator_loss.backward()
            optimizer_G.step()
            print(f"Epoch={epoch}, i={i}, wd={wd}, g_loss={generator_loss}")

        if i % view_samples_interval == 0:
            fig = view_samples(x_fake)
            plt.savefig(f"q3_{epoch}_{i}.png")
