import sys
sys.path.append('/Users/klapidus/emd')

import jetutils
import numpy as np

SAMPLE_SIZE = 400
NUM_EPOCHS = 3

# SAMPLE_SIZE = 4000
# NUM_EPOCHS = 4

NWORKERS = 4

#N_IMAGE_BINS = 24
# N_IMAGE_BINS = 16
# N_IMAGE_BINS = 40
N_IMAGE_BINS = 30
# N_IMAGE_BINS = 12
BIN_WIDTH = 0.4/N_IMAGE_BINS/2
BINS = np.linspace(-0.4, 0.4, num=N_IMAGE_BINS+1)

#LEARNING_RATE = 1.e-5
#LEARNING_RATE = 5.e-4 #decent res
LEARNING_RATE = 1.e-3
WEIGHT_DECAY = 0.0

_mnumber = 1000
def get_jets(vals, rotate=True):
  jets = []
  start = 0
  for i in vals[3]:
      if i > 0 and i < _mnumber:
          jet_pt = vals[0,start]
          jet = vals[0:3,start+1:start+1+int(i)]
          start = start + 1 + int(i)
          jet = jet[:, (jet[0,:] > 0) & (jet[0,:] < _mnumber)]
          jet = jet.transpose()
          yphi_avg = np.average(jet[:, 1:3], weights=jet[:, 0], axis=0)
          jet[:,1:3] -= yphi_avg
          jet[:, 0] /= jet_pt
          if rotate is True:
              jetutils.align_jet_pc_to_pos_phi(jet)
          jets.append(jet)
  return jets

def custom_mask(array, offset):
    return array > 1.e-6


def scale_hist(hist, offset=1.2):
    # print('1', hist)
    # print('sel_indices', sel_indices)
    # print('2', hist)    #
    # sel_indices = hist < 1.e-3
    # hist = np.where(sel_indices, 0.0, np.log(hist + OFFSET)/np.log(1.0 + OFFSET))
    return np.log(hist + offset)/np.log(1.0 + offset)


def unscale_hist(hist, offset=1.2):
    prod_ = hist * np.log(1.0 + offset)
    return np.exp(prod_) - offset


def norm_hist(hist):
    return hist/np.sum(hist)


def scale_back(jet):
    jet = np.add(jet, 1.0)
    jet = np.multiply(jet, 0.5)
    return jet


def calc_girth(jet):
    girth = 0.0
    pt_sum = 0.0
    for eta_bin in range(N_IMAGE_BINS):
        for phi_bin in range(N_IMAGE_BINS):
            eta_val = BINS[eta_bin] + BIN_WIDTH/2.0
            phi_val = BINS[phi_bin] + BIN_WIDTH/2.0
            pt_val = jet[eta_bin, phi_bin]
            dr = np.sqrt(eta_val**2 + phi_val**2)
            girth += pt_val * dr
            pt_sum += pt_val
    if pt_sum > 1.e-3:
        return girth/pt_sum
    else:
        return 0.0