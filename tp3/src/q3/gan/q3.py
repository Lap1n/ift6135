import argparse
import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from q3.gan import utils
from q3.gan.models import generator_models, discriminator_models

sys.path.append("../../../")
import tp3.src.given_code.classify_svhn as classify_svhn
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
parser.add_argument("--discriminator_model", type=int, default=0, help="discriminator_model")
parser.add_argument("--generator_model", type=int, default=0, help="generator_model")
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()

print("########## Setting Up Experiment ######################")
experiment_path = utils.setup_run_folder(args, run_name="gan")
utils.setup_logging(experiment_path)
csv_logger = utils.CsvLogger(filepath = experiment_path)
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
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# CUDA
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generator = generator_models[args.generator_model](latent_dim=args.latent_dim).to(get_device())
discriminator = discriminator_models[args.discriminator_model]().to(get_device())

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

###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################
print("########## Running Main Loop ##########################")
z_sample = Tensor(64, args.latent_dim).normal_()
for epoch in range(args.n_epochs):
    for i, (x_real, y_real) in enumerate(train):
        if cuda:
            x_real = x_real.to(get_device())
        optimizer_D.zero_grad()
        z = Tensor(x_real.size()[0], args.latent_dim).normal_()
        x_fake = generator(z)

        y_real = discriminator(x_real)
        y_fake = discriminator(x_fake)
        gradient_penality = args.lambda_grad_penality * compute_gradient_penality(discriminator, x_real, x_fake)
        wd = - y_real.mean() + y_fake.mean()
        wd_loss =  wd + gradient_penality
        wd_loss.backward()
        optimizer_D.step()


        iteration = epoch * len(train) + i
        # update the generator avec discriminator has trained for "n_critic" steps
        if iteration % args.n_critic == 0:
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            x_fake = generator(z)
            generator_loss = - torch.mean(discriminator(x_fake))
            generator_loss.backward()
            optimizer_G.step()

        if iteration % args.sample_interval == 0:
            x_sample = generator(z_sample)
            utils.make_samples_fig_and_save(x_sample, experiment_path, epoch, iteration)

        if iteration % args.log_interval == 0 :
            csv_logger.write(iteration, epoch, wd.item(), gradient_penality.item(), wd_loss.item(), generator_loss.item())
            print(f"Epoch={epoch}, i={iteration}, wd={wd_loss}, g_loss={generator_loss}")
            utils.log(epoch, iteration, wd_loss, generator_loss)
    utils.save_generator_and_discriminator(experiment_path, g=generator, d=discriminator)
# train for as much as much as possible
# plt D loss
# remove the interpolate
# make critic 5