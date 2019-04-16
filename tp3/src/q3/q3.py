import argparse
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

sys.path.append("../../../")
import tp3.src.given_code.classify_svhn as classify_svhn
import utils

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=10, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--lambda_grad_penality", type=float, default=10, help="WD gradient penality")
parser.add_argument("--log_interval", type=int, default=100, help="print_interval")
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()

print("########## Setting Up Experiment ######################")
experiment_path = utils.setup_run_folder(args, run_name="gan")
utils.setup_logging(experiment_path)
torch.manual_seed(args.seed)

###############################################################################
#
# LOADING & PROCESSING
#
###############################################################################
train, valid, test = classify_svhn.get_data_loader("svhn", args.batch_size)


###############################################################################
#
# MODEL SETUP
#
###############################################################################
print("")
# class Generator(nn.Module):
#     def __init__(self, latent_dim=100):
#         super(Generator, self).__init__()
#
#         # deconv formula : stride * (input -1) + kernel_size  - 2 * padding
#
#         class View(nn.Module):
#             def __init__(self):
#                 super(View, self).__init__()
#
#             def forward(self, x):
#                 return x.view(-1, 512, 3, 3)
#
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 512 * 3 * 3),  # -1xlatent_dim --> -1x512x3x3
#             View(),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
#             # -1x512x3x3 --> -1x512x5x5
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
#             # -1x512x5x5 --> -1x256x9x9
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
#             # -1x256x9x9 --> -1x128x17x17
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=2),
#             # -1x128x9x9 --> -1x3x32x32
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         return self.model(z)
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         class View(nn.Module):
#             def __init__(self):
#                 super(View, self).__init__()
#
#             def forward(self, x):
#                 return x.view(-1, 512 * 3 * 3)
#
#         # conv formula : (input + 2*padding - kernel_size)/stride + 1
#         self.model = nn.Sequential(
#             # -1x3x32x32 --> -1x64x28x28
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             # -1x64x28x28 --> -1x128x24x24
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             # -1x128x24x24--> -1x256x20x20
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             # -1x256x20x20 --> -1x512x16x16
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             # -1x512x16x16 --> -1x512x3x3
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=4, padding=0),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             View(),
#             nn.Linear(512 * 3 * 3, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity



# CUDA
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generator = Generator(latent_dim=args.latent_dim).to(get_device())
discriminator = Discriminator().to(get_device())

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = optim.Adam(generator.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))


# small modif from q1
def compute_gradient_penality(D, x_real, x_fake):
    a = Tensor(x_real.size()[0], 1, 1, 1).uniform_(0,1)
    a.expand_as(x_real)
    a.requires_grad = True
    z = a * x_real + (1 - a) * x_fake
    D_z = D(z)
    grads = autograd.grad(outputs=D_z,
                          inputs=z,
                          grad_outputs=torch.ones(D_z.size(), device=get_device()),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True,)[0]
    gradient_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def get_wd(D, x_real, x_fake, lambda_grad_penality=10):
    return D(x_real).mean() - D(x_fake).mean() - lambda_grad_penality * compute_gradient_penality(D, x_real, x_fake)


###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################
print("########## Running Main Loop ##########################")
for epoch in range(args.n_epochs):
    for i, (x_real, y_real) in enumerate(train):
        if cuda:
            x_real = x_real.to(get_device())
        optimizer_D.zero_grad()
        z = Tensor(x_real.size()[0], args.latent_dim).normal_()
        x_fake = generator(z)

        wd_loss = - get_wd(discriminator, x_real, x_fake, args.lambda_grad_penality)
        wd_loss.backward()
        optimizer_D.step()

        # update the generator avec discriminator has trained for "n_critic" steps
        if i % args.n_critic == 0:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            x_fake = generator(z)
            generator_loss = - torch.mean(discriminator(x_fake))
            generator_loss.backward()
            optimizer_G.step()

        if i % args.sample_interval == 0:
            utils.make_samples_fig_and_save(x_fake, experiment_path, epoch, i)

        if i % args.log_interval == 0 :
            print(f"Epoch={epoch}, i={i}, wd={wd_loss}, g_loss={generator_loss}")
            utils.log(epoch, i, wd_loss, generator_loss)



# train for as much as much as possible
# plt D loss
# remove the interpolate
# make critic 5