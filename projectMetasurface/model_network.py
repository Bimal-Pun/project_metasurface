import torch.nn as nn
import torch

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
class GeneratorNet(nn.Module):
    #constructor of the class GeneratorNet
    def __init__(self, noiseDim, labelDim):
        super(GeneratorNet, self).__init__()
    #definign the passed parameters
        self.noiseDim = noiseDim
        self.labelDim = labelDim
        self._init_modules()
    #fucniton to initiliaze the modules
    def _init_modules(self):
        self.linearLayer = nn.Sequential(
            #linear layer is used so that the
            nn.Linear((self.noiseDim + self.labelDim), 1024),
            #noiseDim + labelDim = 128 +1 = 130 is the dimension of the input and the output will be 1024
            nn.ReLU(),
            #passing through ReLu, converts all negative numbers to zero and only positive number is passed
            nn.Linear(1024, 4*16*64, bias=False),
            # second linear layer where input channel is 1024 and output channel is 4x16x64 = 4096
            nn.BatchNorm1d(4*16*64),
            #
            nn.ReLU()
        )
        #bias is initalized as zero because we do not need it
        self.mainConvLayer = nn.Sequential(
            # first layer where input channel = 64
            # output channel = 64, kernel_size = 5, stride = 2,
            # padding = 1 and bias = false
            nn.ConvTranspose2d(
                in_channels= 64,
                out_channels= 64,
                kernel_size = 5,
                stride= 2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # second layer
            nn.ConvTranspose2d(64, 32, 5, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # third layer
            nn.ConvTranspose2d(32, 16, 5, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # fourth layer
            nn.ConvTranspose2d(16, 1, 5, 2, 1, bias=True),
        )
        #converting the 1 channel output from the last layer to image
        self.conv2d = nn.Conv2d(1, 1, 5, stride=2, padding=0, bias=True, dilation=1, groups=1)
        self.trim = 3

    def forward(self, labelDim, noiseDim):
        # net is 130 and -1  = channel as it is not defined by us
        net = (torch.cat([labelDim, noiseDim], -1)).to(device)
        # while training the linear layer is called first
        net = self.linearLayer(net)
        # reshaping it to -1 = rows (undefined), 64 channels and 4, 16 ,matrix (2D array
        net = net.view(-1, 64, 4, 16)
        #The conv layer is called next
        net = self.mainConvLayer(net)
        #fimnal conv layer
        net = self.conv2d(net)#
        # trimming the  final image to make is 32, 128
        net = net[:, :, self.trim:-self.trim, self.trim:-self.trim]
        return torch.tanh(net)# noramlizing as it makes values between -1 to 1


class DiscriminatorNet(nn.Module):
    def __init__(self, labelDim):
        super(DiscriminatorNet, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=5, stride=2,
            padding=2,
            bias=True,
            dilation=1,
            groups=1)

        self.convLayer = nn.Sequential(
            self.conv2d,
            nn.LeakyReLU(0.2)
            )

        self.linearLayer = nn.Sequential(
            #first linear layer
            nn.Linear(64*16*64 + labelDim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            #second layer
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            #third layer
            nn.Linear(512, 1)
            )

    def forward(self, img, labels):
        # image is passed (either real or fake) with the label values = 2.0

        # adding a small amount of noise to .to(device)the passed image so that the discriminator doesnt overfit

        net = img + (torch.randn(img.size()) * 0.1).to(device, dtype=torch.float)

        net = self.convLayer(net)
        # changing 2d into 1 array so that it can be passed to teh linear layer in self liner layer
        net = net.view(net.size(0), -1)
        net = torch.cat([net, labels], -1)
        net = self.linearLayer(net)
        return net