import datetime
import errno
import os
import pprint
import random

import dateutil.tz
import torch
from shutil import copyfile
import numpy as np

from q3.vae.config import cfg_from_file, cfg


class StatsRecorder:
    """
    Object that wraps all the stats related values. They are saved in the model states, so you can load them again.
    """

    def __init__(self):
        self.train_loss_history = []
        self.valid_best_accuracy = 0.0
        self.valid_losses = []
        self.best_score = -100000000.0
        self.scores = []
        self.generated_images = []


def load_config(args):
    '''
    Load the config .yml file.

    '''

    if args.cfg is None:
        raise Exception("No config file specified.")

    cfg_from_file(args.cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    print('timestamp: {}'.format(timestamp))

    cfg.TIMESTAMP = timestamp
    cfg.OUTPUT_DIR = os.path.join(
        args.results_dir,
        '%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp))

    mkdir_p(cfg.OUTPUT_DIR)
    copyfile(args.cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yml'))

    print('Output dir: {}'.format(cfg.OUTPUT_DIR))

    print('Using config:')
    pprint.pprint(cfg)
    return cfg


def fix_seed(seed):
    '''
    Fix the seed.

    Parameters
    ----------
    seed: int
        The seed to use.

    '''
    print('pytorch/random seed: {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mkdir_p(path):
    '''
    Make a directory.

    Parameters
    ----------
    path : str
        path to the directory to make.

    '''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
