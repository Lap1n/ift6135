import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

def get_config(path):
    config = {}
    f = open(path,'r')
    for config_str in f:
        x = config_str.split("    ")
        config[x[0].strip()] = x[1].strip()
    return config

def get_config_str(config):
    config_str = config['model'] + ", " + config['optimizer'] + ", hidden: " + \
            config['hidden_size'] + ", num_layers: " + config['num_layers'] + \
            ", batch size : " +  config['batch_size'] + ", init lr : " +  \
            config['initial_lr'] + ", keep proba: " + config['dp_keep_prob']
    return config_str
        

def get_wallclocktime(log_path):
    # extract wall clock time from the log file
    wall_clock_time = []
    cumul_time=0
    f = open(log_path,'r')
    for epoch_str in f:
        time_str = epoch_str.split("spent in epoch: ")
        if len(time_str)>1:
            time_spent = float(time_str[1])
            cumul_time += time_spent
            wall_clock_time.append(cumul_time)

    return wall_clock_time

def plot_lc(lc, wct, config, savedir):
    # save as individual figures
    # plot according to epochs
    title = get_config_str(config)
    
    plt.figure(figsize=(10,6))
    plt.plot(lc['train_ppls'][1:], label='train')
    plt.plot(lc['val_ppls'][1:], label='valid')
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("PPL")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(savedir,"ppl_epoch.png"))
    plt.show()
    
    # plot according to wall clock time
    # create time axis 
    wall_clock_time = wct
    
    plt.figure(figsize=(10,6))
    plt.plot(wall_clock_time[1:], lc['train_ppls'][1:], label='train')
    plt.plot(wall_clock_time[1:], lc['val_ppls'][1:], label='valid')
    plt.title(title, fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("PPL")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(savedir,"ppl_time.png"))
    plt.show()
    

if __name__ == "__main__":    
    
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

    matplotlib.rc('font', **font)

    parser = argparse.ArgumentParser(description='Plot learning curve of one experiment')
    
    # folder where the learning curve dict was saved
    parser.add_argument('--dir', type=str, default='',
                        help='directory of learning_curves.npy')
    
    # get learning curve data
    args = parser.parse_args()
    
    # get learning curves
    directory = args.save_dir
    #directory = "C:/Users/alice/Documents/DESS/IFT6135/TP2/ift6135/tp2/src/question3/subquestion4.3/exp5"
    lc_path = os.path.join(directory, 'learning_curves.npy')
    lc = np.load(lc_path)[()]
    
    # wall clock time
    log_path = os.path.join(directory, "log.txt")
    wct = get_wallclocktime(log_path)
    
    # get experiment config
    config_path = os.path.join(directory, "exp_config.txt")
    exp_config = get_config(config_path)
    plot_lc(lc, wct, exp_config, directory)