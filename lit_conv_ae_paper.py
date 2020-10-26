import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

import utils

#exp
# DIMENSION = int(utils.N_IMAGE_BINS/4) #with maxpooling
DIMENSION = utils.N_IMAGE_BINS #without maxpooling
#DIMENSION = 10
# NFC1 = 1024
#NFC2 = 32

class ConvEncoder(nn.Module):
    def __init__(self, hidden_size=16, channels=(8, 16, 32), act=torch.relu):
        super(ConvEncoder, self).__init__()

        self.channels = channels

        #in channels, out channels, kernel_size, stride, padding
        # conv output shape 4D:
        # batch channels img_x img_y
        self.cl1 = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        # self.mp1 = nn.MaxPool2d(2)

        self.cl2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        # self.mp2 = nn.MaxPool2d(2)

        self.cl3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)

        #self.cl3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=0, bias=False)
        #for 32x32 input -> output dimension 6*6*nchannels
        #for 24x24 input -> output dimension 4*4*nchannels
        #for 16x16 input -> output dimension 2*2*nchannels

        #self.cl1_bn = nn.BatchNorm2d(CHANNELS[0])
        #self.cl2_bn = nn.BatchNorm2d(CHANNELS[1])
        #self.cl3_bn = nn.BatchNorm2d(CHANNELS[2])

        # self.fc1 = nn.Linear(DIMENSION*DIMENSION*channels[2], NFC1)
        # self.fc2 = nn.Linear(NFC1, hidden_size)

        self.act = act

    def forward(self, x):
        conv1 = self.cl1(x)
        #print('conv1.shape', conv1.shape)
        conv1 = self.act(conv1)

        #mp1 = self.mp1(conv1)
        #print('mp1.shape', mp1.shape)

        #conv1 = self.cl1_bn(conv1)

        #conv2 = self.cl2(mp1)
        conv2 = self.cl2(conv1)
        #print('conv2.shape', conv2.shape)
        #conv2 = self.cl2_bn(conv2)
        conv2 = self.act(conv2)

        #mp2 = self.mp2(conv2)
        #print('mp2.shape', mp2.shape)

        # hidden = conv2
        # return hidden

        conv3 = self.cl3(conv2)
        #conv3 = self.cl3(mp2)
        #print('conv3 shape', conv3.shape)
        #print('conv3.shape', conv3.shape)
        #print('conv3.shape', conv3.shape)
        #conv3 = self.cl3_bn(conv3)
        conv3 = self.act(conv3)

        hidden = conv3
        return hidden


        #final conv layer = 10x10
        # flat_repr = conv3.view(-1, DIMENSION*DIMENSION*self.channels[2])
        # print('flat.shape', flat_repr.shape)

        # linear1 = self.fc1(flat_repr)
        #print('fc1.shape', linear1.shape)
        # linear1 = self.act(linear1)

        # hidden = linear1

        # linear2 = self.fc2(linear1)
        #print('hidden.shape', linear2.shape)
        # hidden = self.act(linear2)

        # return hidden


class ConvDecoder(nn.Module):
    def __init__(self, hidden_size=16, channels=(8, 16, 32), act=torch.relu):
        super(ConvDecoder, self).__init__()

        self.channels = channels

        # self.fc1 = nn.Linear(hidden_size, NFC1)
        # self.fc2 = nn.Linear(NFC1, DIMENSION*DIMENSION*channels[2])

        self.dcl1 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, padding=1)
        # self.us1 = nn.Upsample(scale_factor=2)

        self.dcl2 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=3, padding=1)
        # self.us2 = nn.Upsample(scale_factor=2)

        self.dcl3 = nn.ConvTranspose2d(channels[0], 1, kernel_size=3, padding=1)

        #self.dcl1_bn = nn.BatchNorm2d(CHANNELS[1])
        #self.dcl2_bn = nn.BatchNorm2d(CHANNELS[0])
        #self.dcl3_bn = nn.BatchNorm2d(CHANNELS[0])

        self.act = act
        self.softmax = nn.Softmax(dim=1) #softmax along second dim

    def forward(self, x):

        # linear1 = self.fc1(x)
        # linear1 = self.act(linear1)
        #print('Decoder fc1.shape', linear1.shape)

        # linear2 = self.fc2(linear1)
        # linear2 = self.fc2(x)
        # linear2 = self.act(linear2)
        # print('Decoder fc2.shape', linear2.shape)

        # x = linear2.view(-1, self.channels[2], DIMENSION, DIMENSION)
        #print('Decoder x.shape', x.shape)

        dconv1 = self.dcl1(x)
        #print('Decoder donv1.shape', dconv1.shape)
        #dconv1 = self.dcl1_bn(dconv1)
        dconv1 = self.act(dconv1)

        #us1 = self.us1(dconv1)
        #print('Decoder us1.shape', us1.shape)

        dconv2 = self.dcl2(dconv1)
        # dconv2 = self.dcl2(us1)
        #print('Decoder dconv2.shape', dconv2.shape)
        #dconv2 = self.dcl2_bn(dconv2)
        dconv2 = self.act(dconv2)

        #us2 = self.us2(dconv2)
        #print('Decoder us2.shape', us2.shape)

        dconv3 = self.dcl3(dconv2)
        # dconv3 = self.dcl3(us2)
        #print('dconv3.shape', dconv3.shape)
        # dconv3 = self.act(dconv3)
        # dconv3 = self.dcl3_bn(dconv3)

        # dconv3 = self.act(dconv3)

        #flatten and softmax the last dconv layer
        flat_view = dconv3.view(-1, 1*utils.N_IMAGE_BINS*utils.N_IMAGE_BINS)
        flat_view =  self.softmax(flat_view)
        output = flat_view.view(-1, 1, utils.N_IMAGE_BINS, utils.N_IMAGE_BINS)

        return output



class LitAE(pl.LightningModule):

    def __init__(self, hidden_size=16, channels=(8, 16, 32), loss=F.mse_loss):
        super().__init__()
        self.encoder = ConvEncoder(hidden_size=hidden_size, channels=channels)
        self.decoder = ConvDecoder(hidden_size=hidden_size, channels=channels)

        self.loss = loss

        # self.loss_fn = F.mse_loss
        # self._loss = None
        # self.optim = optim.Adam(self.parameters(), lr=utils.LEARNING_RATE, weight_decay=utils.WEIGHT_DECAY)

    def forward(self, x):
        #x = x.view(-1, utils.N_IMAGE_BINS, utils.N_IMAGE_BINS)
        x = torch.unsqueeze(x, 1)
        h = self.encoder(x)
        output = self.decoder(h)
        output = torch.squeeze(output, 1)
        return output

    def training_step(self, batch, batch_idx):
        x_pl = batch['pl']
        # print('x shape', x.shape)
        x_dl = batch['dl']
        input = torch.unsqueeze(x_dl, 1)
        hidden = self.encoder(input)
        output = self.decoder(hidden)
        output = torch.squeeze(output, 1)
        # print('output shape', output.shape)
        # loss = F.mse_loss(output, x_pl)
        # loss = F.binary_cross_entropy(output, x_pl)
        # loss = F.l1_loss(output, x)
        loss = self.loss(output, x_pl)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=utils.LEARNING_RATE)
        return optimizer