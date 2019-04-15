import os
import sys

import matplotlib.pyplot as plt
import numpy as np


## FOLDER STUFF
def get_directory_name_with_number(base_path):
    i = 0
    while os.path.exists(base_path + "_" + str(i)):
        i += 1
    return base_path + "_" + str(i)


def setup_run_folder(args, run_name):
    argsdict = args.__dict__
    argsdict['code_file'] = sys.argv[0]
    runs_directory = "runs"
    if not os.path.exists(runs_directory):
        os.makedirs(runs_directory)
    base_path = os.path.join(runs_directory, run_name)
    experiment_path = get_directory_name_with_number(base_path)
    os.mkdir(experiment_path)

    print("Putting log in %s" % experiment_path)
    argsdict['save_dir'] = experiment_path
    with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
        for key in sorted(argsdict):
            f.write(key + '    ' + str(argsdict[key]) + '\n')
    return experiment_path


## SHOW IMAGE STUFF

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
        ax.imshow(img, aspect='equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig


def savefig(experiment_path, fig, epoch, iteration):
    name = os.path.join(experiment_path, f"epoch_{epoch}_iteration{iteration}.png")
    fig.savefig(name)


def make_samples_fig_and_save(samples, experiment_path, epoch, iteration):
    fig = view_samples(samples)
    savefig(experiment_path, fig, epoch, iteration)
