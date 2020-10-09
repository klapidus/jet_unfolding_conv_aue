import torch
import matplotlib.pyplot as plt

from prepare_datasets import jet_dataloader_train, jet_dataloader_test, jetsPL_train, jetsPL_test
#import conv_ae
import conv_ae_paper
import utils

import numpy as np

def train(epoch, models, log=None):
    train_size = len(jetsPL_train)
    for batch_idx, sample_batched in enumerate(training_loader):

        # print('input shape', sample_batched['pl'].shape)
        # print(conv_ae.Encoder(sample_batched['pl']))
        # conv_ae.AE(sample_batched['pl'])

        # print('batch_idx', batch_idx)
        for model in models.values():
            model.optim.zero_grad()
            output = model(sample_batched['pl'])
            loss = model.loss(output, sample_batched['pl'])
            loss.backward()
            model.optim.step()

        if batch_idx % 150 == 0:
            line = 'Train Epoch: {} [{:05d}/{}] '.format(
                epoch, batch_idx * len(sample_batched['pl']), train_size)
            losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])
            print(line + losses)

    else:
        batch_idx += 1
        line = 'Train Epoch: {} [{:05d}/{}] '.format(
            epoch, batch_idx * len(sample_batched['pl']), train_size)
        losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])
        if log is not None:
            for k in models:
                log[k].append(models[k]._loss)
        print(line + losses)


avg_lambda = lambda l: 'loss: {:.4f}'.format(l)
line = lambda i, l: '{}: '.format(i) + avg_lambda(l)


def test(models, loader, log=None):
    test_size = len(jetsPL_test)
    #test_size = loader.__len__()
    #print(test_size)
    test_loss = {k: 0. for k in models}
    with torch.no_grad():
        for sample_batched in loader:
            #output = {k: m(data_out) for k, m in models.items()}
            output = {k: m(sample_batched['pl']) for k, m in models.items()}
            for k, m in models.items():
                test_loss[k] += m.loss(output[k], sample_batched['pl'], reduction='sum').item()  # sum up batch loss

    for k in models:
        test_loss[k] /= (test_size * utils.N_IMAGE_BINS * utils.N_IMAGE_BINS)
        if log is not None:
            log[k].append(test_loss[k])

    lines = '\n'.join([line(k, test_loss[k]) for k in models]) + '\n'
    report = 'Test set:\n' + lines
    print(report)


if __name__ == '__main__':

    net = conv_ae_paper.AE

    # models = {'4': net(4), '8': net(8), '16': net(16)}
    models = {'4': net(4)}
    #models = {'4': net(4), '8': net(8)}
    train_log = {k: [] for k in models}
    test_log = {k: [] for k in models}

    training_loader = jet_dataloader_train
    test_loader = jet_dataloader_test

    for epoch in range(1, 10):
        for model in models.values():
            model.train()
        train(epoch, models, train_log)
        for model in models.values():
            model.eval()
        test(models, test_loader, test_log)

    output_jets = []
    model = models['4']

    # tensors = torch.stack(jetsPL_test)
    for jet in jetsPL_test:
        with torch.no_grad():
            jet_out = model(jet)
            jet_out = torch.squeeze(jet_out, 0)
            jet_out = jet_out.detach().numpy()
            utils.scale_back(jet_out)
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
        ax.matshow(h, interpolation='none')
    plt.show()

# fig = plt.figure(figsize=(10.0, 10.0))
# fig = plt.figure()
#for idx in range(0, utils.N_EFP):
 #   histos = ([el[idx] for el in efps_pl_test], [el[idx] for el in efps_rec])
    # print(efps_pl_test[:, idx].shape)
  #  ax = fig.add_subplot(6, 6, idx+1)
    # bins = np.linspace(0.0, 0.2, 20)
    # plt.hist(histos, bins=10, range=(0.0, 0.2), label=('INPUT','OUTPUT'), alpha=0.8, color=('g','b'))
   # plt.hist(histos, bins=15, label=('input', 'output'), alpha=0.9, color=('b', 'r'))
    #if 0 == idx:
     #   ax.legend()
    # xtitle = 'EFP_' + str(idx+1)
    # plt.xlabel(xtitle)
    # x1, x2, y1, y2 = plt.axis()
    # ax = fig.add_subplot(6, 6, idx + 1)
    # plt.hist(efps_rec[:][idx], bins=40, range=(0.0, 1.0), label='OUTPUT')
    # plt.axis((0.0, 0.05, 0.0, y2))
#plt.show()

# for epoch in range(1, 20):
#     decoder_output(models, test_loader, test_log)

#bins = np.linspace(-0.4, 0.4, num=utils.N_IMAGE_BINS+1)

#model = models['128-2']
#jetDL = jetsDL_test[200]
#h_dl, _, _ = np.histogram2d(jetDL[:, 1], jetDL[:, 2], bins=bins, weights=jetDL[:, 0])
#jetDL = torch.from_numpy(h_pl.astype(np.float32))
#print( model(jetDL) )