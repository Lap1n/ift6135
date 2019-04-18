from __future__ import print_function
# Import comet_ml in the top
from comet_ml import Experiment

import argparse
import os
import sys
import torch
from torch.optim import Adam

from given_code.classify_svhn import get_data_loader
from q3.vae.models.conv_vae import ConvVAE
from q3.vae.models.vae import VAE
from q3.vae.vae_utils import load_config, fix_seed
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
    parser.add_argument("--train_dataset_path", type=str,
                        help='''Train dataset''')
    parser.add_argument("--model", default=None, type=str,
                        help='''If we want to continue training''')
    parser.add_argument("--results_dir", type=str,
                        default='results/',
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
        seed = model_dict["seed"]
    fix_seed(seed)

    hyper_params = cfg.HYPER_PARAMS.INITIAL_VALUES
    train_loader, valid_loader, test_loader = get_data_loader(
        dataset_location=args.train_dataset_path,
        batch_size=cfg.TRAIN.BATCH_SIZE)
    # model = VAE(label=cfg.CONFIG_NAME, image_size=cfg.IMAGE_SIZE,
    #             channel_num=cfg.MODEL.CHANNEL_NUM,
    #             kernel_num=cfg.MODEL.KERNEL_NUM,
    #             z_size=cfg.MODEL.LATENT_SIZE)
    model = ConvVAE(
        width=cfg.IMAGE_SIZE, height=cfg.IMAGE_SIZE,
        nChannels=cfg.MODEL.CHANNEL_NUM,
        hidden_size=500,
        z_dim=cfg.MODEL.LATENT_SIZE, nFilters=cfg.MODEL.KERNEL_NUM
    )

    optimizer = Adam(model.parameters(), lr=hyper_params["LR"])
    trainer = VAETrainer(model=model, optimizer=optimizer, cfg=cfg,
                         train_loader=train_loader, valid_loader=valid_loader,
                         device=device, output_dir=cfg.OUTPUT_DIR,
                         hyper_params=hyper_params,
                         max_patience=cfg.TRAIN.MAX_PATIENCE)
    trainer.fit(hyper_params)
