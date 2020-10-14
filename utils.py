import sys
sys.path.append('/Users/klapidus/emd')

import jetutils
import numpy as np


#N_IMAGE_BINS = 24
# N_IMAGE_BINS = 16
N_IMAGE_BINS = 40
#N_IMAGE_BINS = 20
BIN_WIDTH = 0.4/N_IMAGE_BINS/2

#LEARNING_RATE = 1.e-5
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


def norm_hist_to_max(hist):
    max_ = np.max(hist)
    min_ = np.min(hist)
    #print('hist max', max_)
    #print('hist min', min_)
    if max_ > 0:
        h = (hist - min_) / (max_ - min_) #range 0 - 1
        h = np.divide(h, 0.5) #range (0,2)
        h = np.subtract(h, 1.0)
        #print('hist max', np.max(h))
        #print('hist min', np.min(h))
        return h    #(-1,1)
    else:
        return hist

def scale_back(jet):
    jet = np.add(jet, 1.0)
    jet = np.multiply(jet, 0.5)
    return jet