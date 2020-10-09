import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils

# CHANNELS = (64, 128, 256)
# CHANNELS = (16, 32, 64)
# CHANNELS = (8, 16, 32)
CHANNELS = (4, 8, 16)
# CHANNELS = (2, 4, 8)
DIMENSION = 4

class ConvEncoder(nn.Module):
    def __init__(self, hidden_size=16, act=torch.relu):
        super(ConvEncoder, self).__init__()

        #in channels, out channels, kernel_size, stride, padding
        # conv output shape 4D:
        # batch channels img_x img_y
        self.cl1 = nn.Conv2d(1, CHANNELS[0], kernel_size=4, stride=2, padding=1, bias=False)
        self.cl2 = nn.Conv2d(CHANNELS[0], CHANNELS[1], kernel_size=3, stride=1, padding=0, bias=False)
        self.cl3 = nn.Conv2d(CHANNELS[1], CHANNELS[2], kernel_size=4, stride=2, padding=0, bias=False)
        #for 32x32 input -> output dimension 6*6*nchannels
        #for 24x24 input -> output dimension 4*4*nchannels
        #for 16x16 input -> output dimension 2*2*nchannels

        self.cl1_bn = nn.BatchNorm2d(CHANNELS[0])
        self.cl2_bn = nn.BatchNorm2d(CHANNELS[1])
        self.cl3_bn = nn.BatchNorm2d(CHANNELS[2])

        self.fc = nn.Linear(DIMENSION*DIMENSION*CHANNELS[2], hidden_size)

        self.act = act

    def forward(self, x):
        conv1 = self.cl1(x)
        # print('conv1.shape', conv1.shape)
        conv1 = self.act(conv1)

        conv1 = self.cl1_bn(conv1)

        conv2 = self.cl2(conv1)
        # print('conv2.shape', conv2.shape)
        conv2 = self.cl2_bn(conv2)
        conv2 = self.act(conv2)

        conv3 = self.cl3(conv2)
        # print('conv3.shape', conv3.shape)
        conv3 = self.cl3_bn(conv3)
        conv3 = self.act(conv3)


        #final conv layer = 6x6
        flat_repr = conv3.view(-1, DIMENSION*DIMENSION*CHANNELS[2])
        # print('flat.shape', flat_repr.shape)

        linear = self.fc(flat_repr)
        hidden = self.act(linear)
        # print('hidden.shape', hidden.shape)

        return hidden


class ConvDecoder(nn.Module):
    def __init__(self, hidden_size=16, act=torch.relu):
        super(ConvDecoder, self).__init__()

        self.fc = nn.Linear(hidden_size, DIMENSION*DIMENSION*CHANNELS[2])

        self.dcl1 = nn.ConvTranspose2d(CHANNELS[2], CHANNELS[1], kernel_size=4, stride=2, padding=0, bias=False)
        self.dcl2 = nn.ConvTranspose2d(CHANNELS[1], CHANNELS[0], kernel_size=3, stride=1, padding=0, bias=False)
        self.dcl3 = nn.ConvTranspose2d(CHANNELS[0], 1, kernel_size=4, stride=2, padding=1)

        self.dcl1_bn = nn.BatchNorm2d(CHANNELS[1])
        self.dcl2_bn = nn.BatchNorm2d(CHANNELS[0])
        # self.dcl3_bn = nn.BatchNorm2d(CHANNELS[0])

        self.act = act

    def forward(self, x):

        linear = self.fc(x)
        linear = self.act(linear)
        # print('linear.shape', linear.shape)
        #linear output, reshape to 6x6
        x = linear.view(-1, CHANNELS[2], DIMENSION, DIMENSION)
        # print('x.shape', x.shape)

        # x = torch.unsqueeze(x, 1)

        dconv1 = self.dcl1(x)
        # print('donv1.shape', dconv1.shape)
        dconv1 = self.dcl1_bn(dconv1)
        dconv1 = self.act(dconv1)

        dconv2 = self.dcl2(dconv1)
        # print('donv2.shape', dconv2.shape)
        dconv2 = self.dcl2_bn(dconv2)
        dconv2 = self.act(dconv2)

        dconv3 = self.dcl3(dconv2)
        # print('donv3.shape', dconv3.shape)
        # dconv3 = self.act(dconv3)
        # dconv3 = self.dcl3_bn(dconv3)

        output = torch.tanh(dconv3)

        # print('output.shape', output.shape)

        return output



class AE(nn.Module):
    def __init__(self, hidden_size=16):
        super(AE, self).__init__()
        self.E = ConvEncoder(hidden_size=hidden_size)
        self.D = ConvDecoder(hidden_size=hidden_size)

        self.loss_fn = F.mse_loss
        self._loss = None
        self.optim = optim.Adam(self.parameters(), lr=utils.LEARNING_RATE, weight_decay=1.e-4)

    def forward(self, x):
        x = x.view(-1, utils.N_IMAGE_BINS, utils.N_IMAGE_BINS)
        x = torch.unsqueeze(x, 1)
        h = self.E(x)
        output = self.D(h)
        output = torch.squeeze(output, 1)
        return output

    def decode(self, h):
        with torch.no_grad():
            return self.D(h)

    def loss(self, x, target, **kwargs):
        self._loss = self.loss_fn(x, target, **kwargs)
        return self._loss