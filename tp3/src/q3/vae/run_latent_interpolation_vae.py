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

from given_code.classify_svhn import get_data_loader
from q3.vae.models.conv_vae import ConvVAE
from q3.vae.models.vae import VAE
from q3.vae.vae_utils import load_config, fix_seed, UnNormalize
from q3.vae.vae_trainer import VAETrainer

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


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
        seed = 1134
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
        z = Tensor(100, cfg.MODEL.LATENT_SIZE).normal_()
        x_gen = model.decode(z)
        # x_gen = (x_gen-x_gen.min())/(x_gen.max()-x_gen.min())
        x_gen = un_normalize_transform(x_gen)
        save_image(x_gen, os.path.join(args.results_dir, "test.png"))
