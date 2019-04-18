import gc
import os

from comet_ml import Experiment
from comet_ml import OfflineExperiment

import time
from abc import ABC, abstractmethod

import torch

import copy

import numpy as np

from q3.vae.vae_utils import StatsRecorder


class AbstractTrainer(ABC):
    """
    Abstract class that fits the given model
    """

    def __init__(self, model, optimizer, cfg, train_loader, valid_loader,
                 device, output_dir, hyper_params,
                 max_patience=5):
        """
        :param model: pytorch model
        :param optimizer: pytorch optimizaer
        :param cfg: config instance
        :param train_loader: train data loader
        :param valid_loade: valid data laoder
        :param device: gpu device used (ex: cuda:0)
        :param output_dir: output directory where the model and the results will be located
        :param hyper_params: hyper parameters
        :param max_patience: max number of iteration without seeing improvement in accuracy
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.stats = StatsRecorder()
        self.output_dir = output_dir
        self.best_model = None
        self.hyper_params = hyper_params
        self.max_patience = max_patience
        self.current_patience = 0
        self.epoch = 0
        self.comet_ml_experiment = None
        self.last_checkpoint_filename = None

    def initialize_cometml_experiment(self, hyper_params):
        """
        Initialize the comet_ml experiment (only if enabled in config file)
        :param hyper_params: current hyper parameters dictionary
        :return:
        """
        if self.comet_ml_experiment is None \
                and self.cfg.COMET_ML_UPLOAD is True:
            # Create an experiment
            self.comet_ml_experiment = Experiment(
                api_key="TAOEZkbwnavYnudi3hA9VxBfU",
                project_name="ift6135-tp3", workspace="lap1n")
            if self.comet_ml_experiment.disabled is True:
                # There is problably no internet (in the cluster for example)
                # So we create a offline experiment
                self.comet_ml_experiment = \
                    OfflineExperiment(workspace="lap1n",
                                      project_name="general",
                                      offline_directory=self.output_dir)
            self.comet_ml_experiment.log_parameters(hyper_params)

    def fit(self, current_hyper_params, hyper_param_search_state=None):
        self.initialize_cometml_experiment(current_hyper_params)
        print("# Start training #")
        since = time.time()

        for epoch in range(self.epoch, self.cfg.TRAIN.NUM_EPOCHS, 1):
            self.train(current_hyper_params)
            self.validate()
            self.epoch = epoch
            print(
                '\nEpoch: {}/{}'.format(epoch + 1, self.cfg.TRAIN.NUM_EPOCHS))
            self.print_last_epoch_stats()
            if self.cfg.COMET_ML_UPLOAD is True:
                self.upload_to_comet_ml(self.comet_ml_experiment, epoch)
            if self.early_stopping_check(self.model, hyper_param_search_state):
                break
        time_elapsed = time.time() - since
        print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    @classmethod
    @abstractmethod
    def train(self, current_hyper_params):
        """
        Abstract method for the training
        :param current_hyper_params: current hyper parameters dictionary
        """
        pass

    @classmethod
    def validate(self):
        """
        Validate the model
        :param model: pytorch model
        """
        pass

    @classmethod
    def test(self, model):
        """
        Test the model
        :param model: pytorch model
        """
        pass

    def early_stopping_check(self, model, hyper_param_search_state=None):
        """
        Early stop check
        :param model: pytorch model
        :param current_hyper_params: current hyper parameters dictionary
        :return: True if need to stop. False if continue the training
        """
        last_accuracy_computed = self.stats.scores[-1]
        if self.stats.best_score is None \
                or last_accuracy_computed < self.stats.best_score:
            self.stats.best_score = last_accuracy_computed
            self.best_model = copy.deepcopy(model)
            print('Checkpointing new model...')
            model_filename = self.output_dir + '/checkpoint_{}.pth'.format(
                int(self.stats.best_score))
            self.save_current_best_model(model_filename,
                                         hyper_param_search_state)
            if self.last_checkpoint_filename is not None:
                os.remove(self.last_checkpoint_filename)
            self.last_checkpoint_filename = model_filename
            self.current_patience = 0
        else:
            self.current_patience += 1
            if self.current_patience > self.max_patience:
                return True
        return False

    def compute_loss(self, length_logits, digits_logits, length_labels,
                     digits_labels):
        """
        Multi loss computing function
        :param length_logits: length logits tensor (N x 7)
        :param digits_logits: digits legits tensor (N x 5 x 10)
        :param length_labels: length labels (N x 5 x 1)
        :param digits_labels: length labels tensor (N x 1)
        :return: loss tensor value
        """
        loss = torch.nn.functional.cross_entropy(length_logits, length_labels)
        for i in range(digits_labels.shape[1]):
            loss = loss + torch.nn.functional.cross_entropy(digits_logits[i],
                                                            digits_labels[:,
                                                            i],
                                                            ignore_index=-1)
        return loss

    def load_state_dict(self, state_dict):
        """
        Loads the previous state of the trainer
        Should be overriden in the children classes if needed
        (see LRSchedulerTrainer for an example)
        :param state_dict: state dictionary
        """
        self.epoch = state_dict["epoch"]
        self.stats = state_dict["stats"]
        self.current_patience = state_dict["current_patience"]
        self.best_model = self.model

    def get_state_dict(self, hyper_param_search_state=None):
        """
         Gets the current state of the trainer
         Should be overriden in the children classes if needed
         (see LRSchedulerTrainer for an example)
         :param hyper_param_search_state: hyper param search state if we are
         doing an hyper params serach
         (None by default)
        :return state_dict
         """
        seed = np.random.get_state()[1][0]
        return {
            'epoch': self.epoch + 1,
            'model_state_dict': self.best_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'seed': seed,
            'current_patience': self.current_patience,
            'hyper_param_search_state': hyper_param_search_state,
            'hyper_params': self.hyper_params,
        }

    def save_current_best_model(self, out_path, hyper_param_search_state=None):
        """
        Saves the current best model
        :param out_path: output path string
        :param hyper_param_search_state: hyper param search state if we are
        doing an hyper params serach
        (None by default)
        """
        state_dict = self.get_state_dict(hyper_param_search_state)
        torch.save(state_dict, out_path)
        print("Model saved!")

    def add_plots_summary(self, summary_writer):
        """
        Add plotting values for tensor board
        :param summary_writer: Summary writer object from tensor board
        """
        # plot loss curves
        loss_dict = {'Train loss': self.stats.train_loss_history,
                     'Valid loss': self.stats.valid_losses}
        axis_labels = {'x': "Epochs", 'y': "Loss"}
        summary_writer.plot_curves(loss_dict, "Learning curves", axis_labels)

        # plot accuracy curves
        acc_dict = {'Valid accuracy': self.stats.valid_accuracies,
                    'Length accuracy': self.stats.length_accuracy}
        axis_labels = {'x': "Epochs", 'y': "Accuracy"}
        summary_writer.plot_curves(acc_dict, "Accuracy curves", axis_labels)

    def print_last_epoch_stats(self):
        pass

    def upload_to_comet_ml(self, experiment, epoch):
        """
        Upload to comet_ml the experiment. Only if the flag COMET_ML_UPLOAD is set to true
        :param experiment: comet ml experiment object
        :param epoch: current epoch number
        :return:
        """
        pass
