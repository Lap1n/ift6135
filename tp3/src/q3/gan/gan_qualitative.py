import argparse
import sys
import os

import torch
from torch import Tensor
import numpy as np

from q3.gan.models import discriminator_models, generator_models

sys.path.append("../../")
sys.path.append("../../../")
from torchvision.utils import save_image
from tp3.src.q3.vae.vae_utils import fix_seed, UnNormalize


def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument("--train_dataset_path", type=str,
                        help='''Train dataset''')
    parser.add_argument("--d_model", default=None, type=str,
                        help='''If we want to continue training''')
    parser.add_argument("--g_model", default=None, type=str,
                        help='''If we want to continue training''')
    parser.add_argument("--results_dir", type=str,
                        default='results_gen/gan_qualitative/',
                        help='''results_dir will be the absolute
                        path to a directory where the output of
                        your training will be saved.''')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: {}".format(device))

    d_model_dict = torch.load(args.d_model, map_location=device)
    g_model_dict = torch.load(args.g_model, map_location=device)
    #print(d_model_dict.keys())
    d_model = discriminator_models[0]()
    g_model = generator_models[0]()

    d_model.load_state_dict(d_model_dict)
    g_model.load_state_dict(g_model_dict)


    d_model.eval()
    g_model.eval()

    fix_seed(1344)
    un_normalize_transform = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    ### disentangled representation
    #   perturbation on each dimension
    z = Tensor(1, 100).normal_()
    x_gen = torch.zeros((100, 3, 32, 32))
    epsilon = 2

    for i in range(100):
        perturb = torch.zeros_like(z)
        perturb[0, i] = epsilon
        z_perturb = z + perturb
        x_gen[i] = g_model(z_perturb)
        x_gen[i] = un_normalize_transform(x_gen[i])

    save_image(x_gen, os.path.join(args.results_dir, "perturbations_gan.png"), nrow=10)



    ### interpolation
    #   latent space
    z_0 = Tensor(1, 100).normal_()
    z_1 = Tensor(1, 100).normal_()
    x_gen = torch.zeros((11, 3, 32, 32))

    for i, alpha in enumerate(np.linspace(0,1,11)):
        z_alpha = alpha * z_0 + (1 - alpha) * z_1
        x_gen[i] = g_model(z_alpha)
        x_gen[i] = un_normalize_transform(x_gen[i])
    
    save_image(x_gen, os.path.join(args.results_dir, "latent_interpolation_gan.png"), nrow=11)
