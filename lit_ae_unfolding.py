import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from prepare_datasets import jet_dataloader_train, jet_dataloader_test
from prepare_datasets import jetsPL_train, jetsDL_train
from prepare_datasets import jetsPL_test, jetsDL_test
#import conv_ae
import lit_conv_ae_paper
import utils

from utils import unscale_hist
from utils import calc_girth

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np

import random

import itertools


def inference_jets(jets, trained_model):
    jets_tensor = torch.stack(jets)
    with torch.no_grad():
        jets_out = trained_model(jets_tensor)
        # decoded_jets = jets_out.detach().numpy()
    return jets_out


if __name__ == '__main__':

    # hidden_size = 16, channels = (8, 16, 32), loss = F.mse_loss):

    #hs_s = [8, 16, 32, 64, 128, 256]
    #channels_s = [(8, 16, 32), (32, 16, 8), (128, 64, 32), (32, 32, 32), (8, 8, 8)]
    #loss_functions = {'mse': F.mse_loss, 'bce': F.binary_cross_entropy, 'mae': F.l1_loss}


    hs_s = [-1]
    # channels_s = [(8, 16, 32), (32, 32, 32), (32, 16, 8), (64, 64, 64), (128, 128, 128)]
    # channels_s = [(16, 16, 16)]
    channels_s = [(128, 128, 128)]
    # loss_functions = {'bce': F.binary_cross_entropy} #put softmax in the model
    # loss_functions = {'bce': F.binary_cross_entropy_with_logits} #avoid softmax in the model
    # loss_functions = {'mse': F.mse_loss, 'bce': F.binary_cross_entropy, 'mae': F.l1_loss}
    # loss_functions = {'mse': F.mse_loss, 'bce': F.binary_cross_entropy}
    loss_functions = {'bce': F.binary_cross_entropy}

    test_jet_indices = random.sample(range(0, len(jetsPL_test)), 50)

    train_loader = jet_dataloader_train
    for hs, channels, loss_k in itertools.product(hs_s, channels_s, loss_functions):
        autoencoder = lit_conv_ae_paper.LitAE(hidden_size=hs, channels=channels, loss=loss_functions[loss_k])
        print('hidden_size', hs)
        print('channels', channels)
        print('loss', loss_k)
        #print('loss val', loss_functions[loss_k])
        model_name = f'bin_{utils.N_IMAGE_BINS}_hs_{hs}_chan_{channels}_loss_{loss_k}_ss_{utils.SAMPLE_SIZE}_ne_{utils.NUM_EPOCHS}'
        logdir = f'tb_logs_{loss_k}'
        logger = TensorBoardLogger(logdir, name=model_name)
        trainer = pl.Trainer(max_epochs = utils.NUM_EPOCHS, logger=logger)
        trainer.fit(autoencoder, train_loader)

        output_jets_train = inference_jets(jetsDL_train, autoencoder)
        output_jets_test = inference_jets(jetsDL_test, autoencoder)

        #jet observables
        #pl-dl-dlue

        #scale back, move to numpy arrays
        # jetsPL_test = [unscale_hist(j.detach().numpy()) for j in jetsPL_test]
        # jetsDL_test = [unscale_hist(j.detach().numpy()) for j in jetsDL_test]
        # output_jets = [unscale_hist(j) for j in output_jets]

        # for j in jetsPL_test:
        #     print('pl test integral', np.sum(j.detach().numpy()))

        # for j in jetsDL_test:
        #     print('dl test integral', np.sum(j.detach().numpy()))

        # for j in output_jets:
        #     print('output jet integral', np.sum(j))

        fig = plt.figure(figsize=(10, 10))
        for idx in range(0, 5):
            # jidx = np.random.randint(0, len(jetsPL_test))

            jidx = test_jet_indices[idx]

            jet_pl = jetsPL_test[jidx]
            jet_output = output_jets_test[jidx]
            jet_dl = jetsDL_test[jidx]

            ax = fig.add_subplot(3, 5, idx + 1)
            ax.matshow(jet_pl, interpolation='none')
            ax = fig.add_subplot(3, 5, 5 + idx + 1)
            ax.matshow(jet_output, interpolation='none')
            ax = fig.add_subplot(3, 5, 10 + idx + 1)
            ax.matshow(jet_dl, interpolation='none')
        fig.tight_layout()
        fig_path = f'./{loss_k}_figs/{model_name}'
        plt.savefig(fig_path)

        mass_pl_train = [calc_girth(j.detach().numpy()) for j in jetsPL_train]
        mass_dl_train = [calc_girth(j.detach().numpy()) for j in jetsDL_train]
        mass_output_train = [calc_girth(j.detach().numpy()) for j in output_jets_train]

        mass_pl_test = [calc_girth(j.detach().numpy()) for j in jetsPL_test]
        mass_dl_test = [calc_girth(j.detach().numpy()) for j in jetsDL_test]
        mass_output_test = [calc_girth(j.detach().numpy()) for j in output_jets_test]

        z_pl_test = [calc_z(j.detach().numpy()) for j in jetsPL_test]
        z_dl_test = [calc_z(j.detach().numpy()) for j in jetsDL_test]
        z_output_test = [calc_z(j.detach().numpy()) for j in output_jets_test]

        fig = plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 0.4, 80)
        plt.hist(mass_pl_train, bins, alpha=0.5, label='particle level', color='red')
        plt.hist(mass_dl_train, bins, alpha=0.4, label='detector level', color='green')
        plt.hist(mass_output_train, bins, alpha=0.5, label='NN unfolded', color='blue')
        plt.xlabel('Girth')
        plt.legend(loc='upper right')
        fig_path = f'./{loss_k}_figs/train_girth_{model_name}'
        plt.savefig(fig_path)

        fig = plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 0.4, 80)
        plt.hist2d(mass_pl_test, mass_dl_test, bins)
        plt.xlabel('Girth, Particle Level')
        plt.ylabel('Girth, Detector Level')
        fig_path = f'./{loss_k}_figs/test_girth_pl_dl_{model_name}'
        plt.savefig(fig_path)

        fig = plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 0.4, 80)
        plt.hist2d(mass_pl_test, mass_output_test, bins)
        plt.xlabel('Girth, Particle Level')
        plt.ylabel('Girth, NN unfolded')
        fig_path = f'./{loss_k}_figs/test_girth_pl_output_{model_name}'
        plt.savefig(fig_path)

        fig = plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 0.4, 80)
        plt.hist2d(mass_dl_test, mass_output_test, bins)
        plt.xlabel('Girth, Detector Level')
        plt.ylabel('Girth, NN unfolded')
        fig_path = f'./{loss_k}_figs/test_girth_dl_output_{model_name}'
        plt.savefig(fig_path)

        mse_pl_dl = [F.mse_loss(torch.unsqueeze(j1, 0), torch.unsqueeze(j2, 0)).detach().numpy().item(0)
                     for j1, j2 in zip(jetsPL_test, jetsDL_test)]
        mse_pl_output = [F.mse_loss(torch.unsqueeze(j1, 0), torch.unsqueeze(j2, 0)).detach().numpy().item(0)
                         for j1, j2 in zip(jetsPL_test, output_jets_test)]

        fig = plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 5.e-3, 80)
        plt.hist(mse_pl_dl, bins, alpha=0.5, label='MSE(particle level, detector level)', color='green')
        plt.hist(mse_pl_output, bins, alpha=0.5, label='MSE(particle level, NN unfolded)', color='blue')
        plt.xlabel('MSE')
        plt.legend(loc='upper right')
        fig_path = f'./{loss_k}_figs/test_mse_distr_{model_name}'
        plt.savefig(fig_path)

        bce_pl_dl = [F.binary_cross_entropy(torch.unsqueeze(j1, 0), torch.unsqueeze(j2, 0)).detach().numpy().item(0)
                     for j1, j2 in zip(jetsPL_test, jetsDL_test)]
        bce_pl_output = [F.binary_cross_entropy(torch.unsqueeze(j1, 0), torch.unsqueeze(j2, 0)).detach().numpy().item(0)
                         for j1, j2 in zip(jetsPL_test, output_jets_test)]

        fig = plt.figure(figsize=(8, 6))
        bins = np.linspace(0.0, 0.3, 80)
        plt.hist(bce_pl_dl, bins, alpha=0.5, label='BCE(particle level, detector level)', color='green')
        plt.hist(bce_pl_output, bins, alpha=0.5, label='BCE(particle level, NN unfolded)', color='blue')
        plt.xlabel('BCE')
        plt.legend(loc='upper right')
        fig_path = f'./{loss_k}_figs/test_bce_distr_{model_name}'
        plt.savefig(fig_path)

        #https://discuss.pytorch.org/t/visualize-feature-map/29597/4
        # kernels = autoencoder.encoder.cl1.weight.detach()
        # size_ = int(kernels.size(0)/2)
        # print(size_)
        # fig, axarr = plt.subplots(size_, size_, figsize=(12, 12))
        # for idx1 in range(size_):
        #     for idx2 in range(size_):
        #         axarr[idx1, idx2].imshow(kernels[idx1 + size_*idx2].squeeze())
        # fig_path = f'./{loss_k}_figs/kernels_first_layer_{model_name}'
        # plt.tight_layout()
        # plt.savefig(fig_path)