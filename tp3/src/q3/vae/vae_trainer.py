import os

import torch
from PIL import Image

from tqdm import tqdm
import scipy.misc

from q3.vae.abstract_trainer import AbstractTrainer
import matplotlib.pyplot as plt

from q3.vae.vae_utils import UnNormalize
from torchvision.utils import save_image

GENERATED_FILENAME = "generated.png"


class VAETrainer(AbstractTrainer):
    """
    BaseTrainer class that fits the given model
    """

    def __init__(self, model, optimizer, cfg, train_loader, valid_loader,
                 device, output_dir, hyper_params,
                 max_patience):
        """
        :param model: pytorch model
        :param optimizer: pytorch optimizaer
        :param cfg: config instance
        :param train_loader: train data loaader
        :param valid_loade: valid data laoder
        :param device: gpu device used (ex: cuda:0)
        :param output_dir: output directory where the model and the results
         will be located
        :param hyper_params: hyper parameters
        :param max_patience: max number of iteration without seeing improvement
        in accuracy
        """
        super(VAETrainer, self).__init__(model, optimizer, cfg, train_loader,
                                         valid_loader, device,
                                         output_dir,
                                         hyper_params, max_patience)
        self.un_normalize = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def train(self, current_hyper_params):
        """
        Method for the training
        :param current_hyper_params: current hyper parameters dictionary
        """
        self.model.train()
        n_iter = 0
        total_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(self.train_loader)):
            _, _, total_loss, n_iter = self.train_batch(data, total_loss,
                                                        n_iter)
        self.stats.train_loss_history.append(total_loss / n_iter)

    def validate(self):
        self.model.eval()
        n_iter = 0
        total_valid_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(self.train_loader)):
            x_reconstructed, x, total_valid_loss, n_iter = self.train_batch(
                data,
                total_valid_loss,
                n_iter)

        valid_loss = total_valid_loss / n_iter
        self.stats.valid_losses.append(valid_loss)
        self.stats.scores.append(valid_loss)
        self.save_image(x_reconstructed, x, GENERATED_FILENAME)

    def save_image(self, x_recons, x, filename, n=5):
        x_recons = self.un_normalize(x_recons)
        x = self.un_normalize(x)
        images = torch.stack([x[0: n], x_recons[0: n]]).view(-1, x.shape[1],
                                                             x.shape[2],
                                                             x.shape[3])
        save_image(images,
                   os.path.join(self.output_dir, filename), nrow=n)

    def train_batch(self, x, train_loss, train_n_iter):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        x_reconstructed, mu, logvar = self.model(x)
        loss = self.loss_function(x_reconstructed, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        self.optimizer.step()
        train_n_iter += 1
        return x_reconstructed, x, train_loss, train_n_iter

    def loss_function(self, x_gen, x, mean, logvar):
        reconstruction_loss = self.reconstruction_loss(x_gen, x)
        kl_divergence_loss = self.kl_divergence_loss(mean, logvar)
        return reconstruction_loss + kl_divergence_loss

    def print_last_epoch_stats(self):
        print('\tTrain Loss: {:.4f}'.format(
            self.stats.train_loss_history[-1]))
        print('\tValid Loss: {:.4f}'.format(self.stats.valid_losses[-1]))

    def upload_to_comet_ml(self, experiment, epoch):
        """
        Upload to comet_ml the experiment. Only if the flag COMET_ML_UPLOAD is
        set to true
        :param experiment: comet ml experiment object
        :param epoch: current epoch number
        :return:
        """
        experiment.log_metric("Train loss",
                              self.stats.train_loss_history[-1],
                              step=epoch)
        experiment.log_metric("Valid loss", self.stats.valid_losses[-1],
                              step=epoch)
        experiment.log_image(
            os.path.join(self.output_dir, GENERATED_FILENAME),
            file_name=None, overwrite=False)

    @staticmethod
    def reconstruction_loss(x_reconstructed, x):
        # return torch.nn.BCELoss(size_average=False)(x_reconstructed,
        #                                             x) / x.size(0)
        return torch.nn.MSELoss(size_average=False)(x_reconstructed,
                                                    x) / x.size(0)

    @staticmethod
    def kl_divergence_loss(mean, logvar):
        return ((
                        mean ** 2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(
            0)
