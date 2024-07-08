import torch
from torch import nn
#import torch.nn.functional as Func
import torch.nn.functional as F

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True, res_num=1):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias, res_num=res_num)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, res_num):
        super(ResnetBlock, self).__init__()

        self.res_num = res_num
        self.stem = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            ) for i in range(res_num)
        ])
    def forward(self, x):

        if self.res_num > 1:
            temp = x

        for i in range(self.res_num):
            xx = self.stem[i](x)
            x = x + xx
            x = F.leaky_relu(x, 0.1, inplace=True)

        if self.res_num > 1:
            x = x + temp

        return x
def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True, act='LeakyReLU'):
    if act is not None:
        if act == 'LeakyReLU':
            act_ = nn.LeakyReLU(0.1,inplace=True)
        elif act == 'Sigmoid':
            act_ = nn.Sigmoid()
        elif act == 'Tanh':
            act_ = nn.Tanh()

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
            act_
        )
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)

class IFAN_module(nn.Module):
    '''
    Here is the implementation of IFAN modeule, which include the following three modules:
        1. filter ecoder, take input image into filter result; 
        2. Disparity Map Estimator, estimate the Disparity Map;
        3. Filter predictor, predict the Deblur mapping for condition; 
    '''
    def __init__(self, config):
        super(IFAN_module, self).__init__()
        #config = config.IFAN
        ch1 = config.ch
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 4
        self.ch4 = ch4
        # filter encoder 
        ks = config.ks
        self.kconv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.kconv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.kconv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.kconv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.kconv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.kconv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.kconv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)
        res_num = config.res_num
        # add disparity map estimator 
        self.DME = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, 1, kernel_size=3, act=None)
        )
        # # add filter predctor
        # self.conv_DME = conv(1, ch4, kernel_size=3)
        # self.N = config.N
        # self.kernel_dim = self.N * (ch4)
        self.conv_DME = conv(1, ch4, kernel_size=3)
        self.Fun = nn.Sequential(
            conv(ch4, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks, res_num=res_num),
            resnet_block(ch3, kernel_size=ks, res_num=res_num),
            nn.PixelShuffle(8),
            conv(2, 1, kernel_size=1, act = None))
#             nn.PixelShuffle(4),
#             conv(ch1, 3, kernel_size=1, act = None))
    def forward(self, C):
        '''
            Given combined image C:
        '''
        f = self.kconv1_3(self.kconv1_2(self.kconv1_1(C)))
        f = self.kconv2_3(self.kconv2_2(self.kconv2_1(f)))
        f = self.kconv3_3(self.kconv3_2(self.kconv3_1(f)))
        f = self.kconv4_3(self.kconv4_2(self.kconv4_1(f)))
        # pass disparity map estimator
        DM = self.DME(f)
        # print(self.conv_DME(DM).shape)
        DM_3 = self.Fun(self.conv_DME(DM))
        return DM_3