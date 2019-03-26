# Generates all the required plots for problem 4
# Folder name convention : MODEL_OPTIMIZER_changedparams

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from learning_curve import get_wallclocktime, get_config

def plot_architecture(plot_info, title):
    # over epoch
    fig,axs = plt.subplots(2,1, figsize=(10,8))
    fig.suptitle(title)
    
    for curve in plot_info:
        axs[0].plot(curve['ppl'])
        axs[1].plot(curve['time'], curve['ppl'], 
                 label=curve['config']['optimizer'] + ",id:" + curve['id'] )
    
    axs[1].legend(fontsize=12)
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('PPL')
    axs[1].set_xlabel('time (s)')
    axs[0].grid(True)
    axs[1].grid(True)
    
    plt.savefig(title + ".png")
    
def plot_optimizer(plot_info, title):
    fig,axs = plt.subplots(2,1, figsize=(10,8))
    fig.suptitle(title)
    
    for curve in plot_info:
        axs[0].plot(curve['ppl'])
        axs[1].plot(curve['time'], curve['ppl'], 
                 label=curve['config']['model'] + ",id:" + curve['id'] )
    
    axs[1].legend(fontsize=12)
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('PPL')
    axs[1].set_xlabel('time (s)')
    axs[0].grid(True)
    axs[1].grid(True)
    
    plt.savefig(title + ".png")
    

parser = argparse.ArgumentParser(description='Problem 4 - plots')

# folder where the all the experiment folders are -- these need to keep the 
# name convention in the ptb-lm script
parser.add_argument('--dir', type=str, default='',
                    help='directory of experiment folders')

args = parser.parse_args()

#workdir = "results/"
workdir = os.getcwd()
if args.dir:
    workdir = args.dir

# optimizers
VSGD = []
SGDlrs = []
Adam = []

# architectures
RNN = []
GRU = []
Transformer = []

subdirs = [x[0] for x in os.walk(workdir)]                                                                            
for i, subdir in enumerate(subdirs,0): 
    if (i != 0): 
        params = get_config(os.path.join(subdir, 'exp_config.txt'))
        # get the learning_curve.npy file
        lc_path = os.path.join(subdir, 'learning_curves.npy')
        lc = np.load(lc_path)[()]
        
        # get wall clock time 
        wall_clock_time = get_wallclocktime(os.path.join(subdir, 'log.txt'))
        
        
        
        # get plot info for the type of architecture
        plot_info = {'time': wall_clock_time,
                     'ppl': lc['val_ppls'],
                     'config' : params,
                     'id': str(i)}
    
        if params['model'] == 'RNN':
            RNN.append(plot_info)
        if params['model'] == 'GRU':
            GRU.append(plot_info)
        if params['model'] == 'TRANSFORMER':
            Transformer.append(plot_info)
       
        
        # get the type of optimizer
        if params['optimizer'] == 'ADAM':
            Adam.append(plot_info)
        if params['optimizer'] == 'SGD_LR_SCHEDULE':
            SGDlrs.append(plot_info)
        if params['optimizer'] == 'SGD':
            VSGD.append(plot_info)
            
      
        
# plot validation curves for each architecture, over epoch and time
plot_architecture(RNN, "Validation curves for RNN")
plot_architecture(GRU, "Validation curves for GRU")
plot_architecture(Transformer, "Validation curves for Transformer")

# plot validation curves for each optimizer, over epoch and time
plot_optimizer(Adam, "Validation curves for Adam optimizer")
plot_optimizer(SGDlrs, "Validation curves for SGD with scheduled learning rate")
plot_optimizer(VSGD, "Validation curves for vanilla SGD.")
