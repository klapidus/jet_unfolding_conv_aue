import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils

# CHANNELS = (64, 128, 256)
# CHANNELS = (16, 32, 64)
# CHANNELS = (8, 16, 32)
#CHANNELS = (128, 8, 16)
CHANNELS = (128, 128, 128)
DIMENSION = 10
NFC1 = 32
NFC2 = 6

class ConvEncoder(nn.Module):
    def __init__(self, hidden_size=16, act=torch.relu):
        super(ConvEncoder, self).__init__()

        #in channels, out channels, kernel_size, stride, padding
        # conv output shape 4D:
        # batch channels img_x img_y
        self.cl1 = nn.Conv2d(1, CHANNELS[0], kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(2)

        self.cl2 = nn.Conv2d(CHANNELS[0], CHANNELS[1], kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(2)

        self.cl3 = nn.Conv2d(CHANNELS[1], CHANNELS[2], kernel_size=3, padding=1)

        #self.cl3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=0, bias=False)
        #for 32x32 input -> output dimension 6*6*nchannels
        #for 24x24 input -> output dimension 4*4*nchannels
        #for 16x16 input -> output dimension 2*2*nchannels

        #self.cl1_bn = nn.BatchNorm2d(CHANNELS[0])
        #self.cl2_bn = nn.BatchNorm2d(CHANNELS[1])
        #self.cl3_bn = nn.BatchNorm2d(CHANNELS[2])

        self.fc1 = nn.Linear(DIMENSION*DIMENSION*CHANNELS[2], NFC1)

        self.fc2 = nn.Linear(NFC1, NFC2)

        self.act = act

    def forward(self, x):
        conv1 = self.cl1(x)
        #print('conv1.shape', conv1.shape)
        conv1 = self.act(conv1)

        mp1 = self.mp1(conv1)
        #print('mp1.shape', mp1.shape)

        #conv1 = self.cl1_bn(conv1)

        conv2 = self.cl2(mp1)
        #print('conv2.shape', conv2.shape)
        #conv2 = self.cl2_bn(conv2)
        conv2 = self.act(conv2)

        mp2 = self.mp2(conv2)
        #print('mp2.shape', mp2.shape)

        conv3 = self.cl3(mp2)
        #print('conv3.shape', conv3.shape)
        # print('conv3.shape', conv3.shape)
        #conv3 = self.cl3_bn(conv3)
        conv3 = self.act(conv3)


        #final conv layer = 10x10
        flat_repr = conv3.view(-1, DIMENSION*DIMENSION*CHANNELS[2])
        #print('flat.shape', flat_repr.shape)

        linear1 = self.fc1(flat_repr)
        #print('fc1.shape', linear1.shape)
        linear1 = self.act(linear1)

        linear2 = self.fc2(linear1)
        #print('hidden.shape', linear2.shape)
        hidden = self.act(linear2)

        return hidden


class ConvDecoder(nn.Module):
    def __init__(self, hidden_size=16, act=torch.relu):
        super(ConvDecoder, self).__init__()

        self.fc1 = nn.Linear(NFC2, NFC1)
        self.fc2 = nn.Linear(NFC1, DIMENSION*DIMENSION*CHANNELS[2])

        self.dcl1 = nn.ConvTranspose2d(CHANNELS[2], CHANNELS[1], kernel_size=3, padding=1)
        self.us1 = nn.Upsample(scale_factor=2)

        self.dcl2 = nn.ConvTranspose2d(CHANNELS[1], CHANNELS[0], kernel_size=3, padding=1)
        self.us2 = nn.Upsample(scale_factor=2)

        self.dcl3 = nn.ConvTranspose2d(CHANNELS[0], 1, kernel_size=3, padding=1)
        #self.us3 = nn.Upsample(scale_factor=2)

        #self.dcl1_bn = nn.BatchNorm2d(CHANNELS[1])
        #self.dcl2_bn = nn.BatchNorm2d(CHANNELS[0])
        #self.dcl3_bn = nn.BatchNorm2d(CHANNELS[0])

        self.act = act
        self.act_softmax = nn.Softmax()

    def forward(self, x):

        linear1 = self.fc1(x)
        linear1 = self.act(linear1)
        #print('Decoder fc1.shape', linear1.shape)

        linear2 = self.fc2(linear1)
        linear2 = self.act(linear2)
        #print('Decoder fc2.shape', linear2.shape)

        #linear output, reshape to 6x6
        x = linear2.view(-1, CHANNELS[2], DIMENSION, DIMENSION)
        #print('Decoder x.shape', x.shape)

        # x = torch.unsqueeze(x, 1)

        dconv1 = self.dcl1(x)
        #print('Decoder donv1.shape', dconv1.shape)
        #dconv1 = self.dcl1_bn(dconv1)
        dconv1 = self.act(dconv1)

        us1 = self.us1(dconv1)
        #print('Decoder us1.shape', us1.shape)


        dconv2 = self.dcl2(us1)
        #print('Decoder dconv2.shape', dconv2.shape)
        #dconv2 = self.dcl2_bn(dconv2)
        dconv2 = self.act(dconv2)

        us2 = self.us2(dconv2)
        #print('Decoder us2.shape', us2.shape)

        dconv3 = self.dcl3(us2)
        #print('dconv3.shape', dconv3.shape)
        #dconv3 = self.act(dconv3)
        # dconv3 = self.dcl3_bn(dconv3)

        output = torch.tanh(dconv3)

        #TODO CLARIFY WITH SOFTMAX
        #output = dconv3.view(-1, 1, 1600)
        #print('output.shape', output.shape)
        #act_softmax =
        #output = self.act_softmax(output)
        #decoded = output.view(-1, 1, 40, 40)
        #print('decoded.shape', decoded.shape)

        return output



class AE(nn.Module):
    def __init__(self, hidden_size=16):
        super(AE, self).__init__()
        self.Encoder = ConvEncoder(hidden_size=hidden_size)
        self.Decoder = ConvDecoder(hidden_size=hidden_size)

        self.loss_fn = F.mse_loss
        self._loss = None
        self.optim = optim.Adam(self.parameters(), lr=utils.LEARNING_RATE, weight_decay=1.e-4)

    def forward(self, x):
        x = x.view(-1, utils.N_IMAGE_BINS, utils.N_IMAGE_BINS)
        x = torch.unsqueeze(x, 1)
        h = self.Encoder(x)
        output = self.Decoder(h)
        output = torch.squeeze(output, 1)
        return output

    def decode(self, h):
        with torch.no_grad():
            return self.D(h)

    def loss(self, x, target, **kwargs):
        self._loss = self.loss_fn(x, target, **kwargs)
        return self._loss