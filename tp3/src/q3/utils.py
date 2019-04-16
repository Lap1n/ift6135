import os
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np
import logging
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

## FOLDER STUFF
def get_directory_name_with_number(base_path):
    i = 0
    while os.path.exists(base_path + "_" + str(i)):
        i += 1
    return base_path + "_" + str(i)


def setup_run_folder(args, run_name):
    argsdict = args.__dict__
    argsdict['code_file'] = sys.argv[0]
    argsdict['device'] = get_device()
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
    plt.close()


###########
# Save model
###########
def save_model(model, model_name, experiment_path):
    torch.save(model.state_dict(), os.path.join(experiment_path, f'{model_name}.pt'))

def save_generator_and_discriminator(experiment_path, g, d):
    save_model(g, "generator", experiment_path)
    save_model(d, "discriminator", experiment_path)

###########
# LOGGING
###########
def setup_logging(experiment_path):
    filename = os.path.join(experiment_path, 'logs.log')
    logging.basicConfig(filename=filename, format='%(asctime)s - %(message)s', level=logging.INFO)

def log(epoch, iteration, wd, g_loss):
    logging.info(f"epoch={epoch}, i={iteration}, wd={wd}, g_loss={g_loss}")

#CSV LOGGER
class CsvLogger:
    def __init__(self, filepath='./', filename='results.csv', data=None):
        self.log_path = filepath
        self.log_name = filename
        self.csv_path = os.path.join(self.log_path, self.log_name)
        self.fieldsnames = ['epoch', 'iteration', 'wd', 'gradient_penality', "wd_loss", "generator_loss"]

        with open(self.csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writeheader()

        self.data = {}
        for field in self.fieldsnames:
            self.data[field] = []
        if data is not None:
            for d in data:
                d_num = {}
                for key in d:
                    d_num[key] = float(d[key]) if key != 'epoch' else int(d[key])
                    self._write(d_num)

    def write(self, epoch, iteration, wd, gradient_penality, wd_loss, generator_loss):
        self._write({"epoch":epoch,
                    "iteration":iteration,
                    "wd":wd,
                    "gradient_penality":gradient_penality,
                    "wd_loss":wd_loss,
                    "generator_loss":generator_loss})

    def _write(self, data):
        for k in self.data:
            self.data[k].append(data[k])
        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writerow(data)

