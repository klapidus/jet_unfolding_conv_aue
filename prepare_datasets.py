import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import scipy

import matplotlib.pyplot as plt

import utils

class JetPicturesDataset(Dataset):
    def __init__(self, jets_in, jets_out):
        self.jets_in = jets_in
        self.jets_out = jets_out

    def __len__(self):
        return len(self.jets_in)

    def __getitem__(self, idx):
        sample = {'pl': self.jets_in[idx], 'dl': self.jets_out[idx]}
        return sample


_jetsPL = np.load('jet_pl_dluebs_input.npy', allow_pickle=True)
_jetsDL = np.load('jet_dl_dluebs_input.npy', allow_pickle=True)

_jetsPL = _jetsPL[:utils.SAMPLE_SIZE]
_jetsDL = _jetsDL[:utils.SAMPLE_SIZE]

print('PL size = ', len(_jetsPL))
print('DL size = ', len(_jetsDL))

jetsPL_train = []
jetsDL_train = []
jetsPL_test = []
jetsDL_test = []

weight_train = []
weight_test = []

# jet_images_pl_test = []
# jet_images_dl_test = []

for idx, _ in enumerate(_jetsPL):
    jet_pl = _jetsPL[idx]
    jet_dl = _jetsDL[idx]
    h_pl, _, _ = np.histogram2d(jet_pl[:, 1], jet_pl[:, 2], bins=utils.BINS, weights=jet_pl[:, 0])
    h_dl, _, _ = np.histogram2d(jet_dl[:, 1], jet_dl[:, 2], bins=utils.BINS, weights=jet_dl[:, 0])

    if np.sum(h_pl) < 1.e-3 or np.sum(h_dl) < 1.e-3:
        continue

    # h_pl = utils.norm_hist_to_max(h_pl)
    # h_dl = utils.norm_hist_to_max(h_dl)

    # h_pl = utils.scale_hist(h_pl)
    # h_dl = utils.scale_hist(h_dl)

    # girth = utils.calc_girth(h_pl)
    # weight = np.array(girth**2)

    if np.random.randint(2) == 1:
        #tensor_ = torch.from_numpy(h_pl.astype(np.float32))
        jetsPL_train.append(torch.from_numpy(h_pl.astype(np.float32)))
        jetsDL_train.append(torch.from_numpy(h_dl.astype(np.float32)))
        # weight_train.append(torch.from_numpy(weight.astype(np.float32)))
    else:
        jetsPL_test.append(torch.from_numpy(h_pl.astype(np.float32)))
        jetsDL_test.append(torch.from_numpy(h_dl.astype(np.float32)))
        # weight_test.append(torch.from_numpy(weight.astype(np.float32)))
        # jet_images_pl_test.append(h_pl)
        # jet_images_dl_test.append(h_dl)


#fig = plt.figure(figsize=(30, 30))
#for idx, _ in enumerate(_jetsPL):
#    if idx > 24:
#        break
#    jet_pl = _jetsPL[idx]
    # jet_dl = _jetsDL[idx]
#    h_pl, _, _ = np.histogram2d(jet_pl[:, 1], jet_pl[:, 2], bins=BINS, weights=jet_pl[:, 0])
    # h_dl, _, _ = np.histogram2d(jet_dl[:, 1], jet_dl[:, 2], bins=BINS, weights=jet_dl[:, 0])

    # h_plnp.divide(hist, np.amax(hist))
    # print(h_pl.shape)

    # print(h_pl)
    # ax = fig.add_subplot(5, 5, idx + 1)
    #ax.matshow(h_dl, interpolation='none')
    # ax = fig.add_subplot(5, 5, idx + 1)
    # ax.matshow(h_pl, interpolation='none')
# plt.show()


jet_dataset_train = JetPicturesDataset(jetsPL_train, jetsDL_train)
jet_dataset_test = JetPicturesDataset(jetsPL_test, jetsDL_test)

jet_dataloader_train = DataLoader(jet_dataset_train, batch_size=128, shuffle=False, num_workers=utils.NWORKERS)
jet_dataloader_test = DataLoader(jet_dataset_test, batch_size=128, shuffle=False, num_workers=utils.NWORKERS)