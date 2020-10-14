import torch
import matplotlib.pyplot as plt

from prepare_datasets import jet_dataloader_train, jet_dataloader_test, jetsPL_train, jetsPL_test, jet_images_pl_test
#import conv_ae
import lit_conv_ae_paper
import utils

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np

if __name__ == '__main__':

    ae = lit_conv_ae_paper.LitAE()

    train_loader = jet_dataloader_train

    logger = TensorBoardLogger('tb_logs', name='my_model')

    trainer = pl.Trainer(max_epochs = 4, logger=logger)
    trainer.fit(ae, train_loader)

    output_jets = []
    # tensors = torch.stack(jetsPL_test)
    for jet in jetsPL_test:
        with torch.no_grad():
            # batch with size 0:
            jet = torch.unsqueeze(jet, 0)
            jet_out = ae(jet)
            jet_out = torch.squeeze(jet_out, 0)
            jet_out = jet_out.detach().numpy()
            # jet_out = utils.scale_back(jet_out)
            output_jets.append(jet_out)

    fig = plt.figure(figsize=(30, 30))
    for idx in range(0, 34):
        jet = output_jets[idx].copy()
        # jet = jetsPL_test[idx]
        # print(jet)
        h, _, _ = np.histogram2d(jet[:, 1], jet[:, 2], bins=utils.N_IMAGE_BINS, weights=jet[:, 0])
        ax = fig.add_subplot(6, 6, idx + 1)
        # ax.matshow(jet, interpolation='none')
        # ax = fig.add_subplot(5, 5, idx + 1)

        # h = jet_images_pl_test[idx]

        ax.matshow(h, interpolation='none')
    plt.show()