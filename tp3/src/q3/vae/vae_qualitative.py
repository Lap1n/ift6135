from __future__ import print_function
# Import comet_ml in the top
from comet_ml import Experiment

import argparse
import os
import sys
import torch
from torch import Tensor
from torch.optim import Adam
from torchvision.utils import save_image
import numpy as np

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
sys.path.append("../../../../")
sys.path.append("../../../")
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")


print(dir_path)

from given_code.classify_svhn import get_data_loader
from q3.vae.models.conv_vae import ConvVAE
from q3.vae.models.vae import VAE
from q3.vae.vae_utils import load_config, fix_seed, UnNormalize
from q3.vae.vae_trainer import VAETrainer



def parse_args():
    '''
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    '''
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument('--cfg', type=str,
                        default=None,
                        help='''optional config file,
                             e.g. config/base_config.yml''')
    parser.add_argument("--train_dataset_path", type=str,
                        help='''Train dataset''')
    parser.add_argument("--model", default=None, type=str,
                        help='''If we want to continue training''')
    parser.add_argument("--results_dir", type=str,
                        default='results_gen/',
                        help='''results_dir will be the absolute
                        path to a directory where the output of
                        your training will be saved.''')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("pytorch version {}".format(torch.__version__))
    # Load the config file
    cfg = load_config(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: {}".format(device))

    # Make the results reproductible
    seed = cfg.SEED
    model_dict = None
    if args.model is not None:
        model_dict = torch.load(args.model, map_location=device)
        seed = 3134
        fix_seed(seed)
        hyper_params = cfg.HYPER_PARAMS.INITIAL_VALUES
        un_normalize_transform = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        model = ConvVAE(
            width=cfg.IMAGE_SIZE, height=cfg.IMAGE_SIZE,
            nChannels=cfg.MODEL.CHANNEL_NUM,
            hidden_size=500,
            z_dim=cfg.MODEL.LATENT_SIZE,
            nFilters=cfg.MODEL.KERNEL_NUM
        )
        model.load_state_dict(model_dict["model_state_dict"])
        model.eval()

        ### disentangled representation
        #   perturbation on each dimension
        z = Tensor(1, cfg.MODEL.LATENT_SIZE).normal_()
        x_gen = torch.zeros((cfg.MODEL.LATENT_SIZE, 3, 32, 32))
        epsilon = 2

        for i in range(cfg.MODEL.LATENT_SIZE):
            perturb = torch.zeros_like(z)
            perturb[0, i] = epsilon
            z_perturb = z + perturb
            x_gen[i] = model.decode(z_perturb)
            x_gen[i] = un_normalize_transform(x_gen[i])

        save_image(x_gen, os.path.join(cfg.OUTPUT_DIR, "perturbations_vae.png"), nrow=10)


        ### interpolation
        #   latent space
        z_0 = Tensor(1, cfg.MODEL.LATENT_SIZE).normal_()
        z_1 = Tensor(1, cfg.MODEL.LATENT_SIZE).normal_()
        x_gen = torch.zeros((11, 3, 32, 32))

        for i, alpha in enumerate(np.linspace(0,1,11)):
            z_alpha = alpha * z_0 + (1 - alpha) * z_1
            x_gen[i] = model.decode(z_alpha)
            x_gen[i] = un_normalize_transform(x_gen[i])
        
        save_image(x_gen, os.path.join(cfg.OUTPUT_DIR, "latent_interpolation_vae.png"), nrow=11)

        #   data space

        train_loader, valid_loader, test_loader = get_data_loader(
            dataset_location=args.train_dataset_path,
            batch_size=2)

        x_0 = None
        x_1 = None

        for batch_idx, (data, _) in enumerate(train_loader):
            x_0 = data[0]
            x_1 = data[1]
            break
        
        x_gen = torch.zeros((11, 3, 32, 32))

        for i, alpha in enumerate(np.linspace(0,1,11)):
            x_alpha = alpha * x_0 + (1 - alpha) * x_1
            x_gen[i] = x_alpha
        
        save_image(x_gen, os.path.join(cfg.OUTPUT_DIR, "data_interpolation_vae.png"), nrow=11)
