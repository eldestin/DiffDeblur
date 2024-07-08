import torch
import torch.nn as nn

class CL_FFT_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(CL_FFT_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        X = torch.fft.rfft(X)
        Y = torch.fft.rfft(Y)
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class dc_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(dc_loss, self).__init__()
        self.eps = 1e-6
        self.l1 = nn.L1Loss()
    def DC(self,x,patch_size):
        maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
        dc = maxpool(0-x[:, None, :, :, :])
        return dc
    def forward(self, X, Y):
        DC_x = self.DC(X,35)
        DC_y = self.DC(Y,35)
        return self.l1(DC_x,DC_y)

class CC_loss(torch.nn.Module):
    def __init__(self):
        super(CC_loss, self).__init__()
        self.eps = 1e-6
        self.l1 = nn.L1Loss()
    def CC(self,x,patch_size):
        maxpool1 = nn.MaxPool3d((3, 1, 1), stride=1, padding=(0, 1//2, 1//2))
        maxpool2 = nn.MaxPool3d((1,patch_size, patch_size), stride=1)
        x = maxpool1(0-x[:, None, :, :, :])
        cc = maxpool2(0-x)
        return cc
    def forward(self, X, Y):
        DC_x = self.CC(X,35)
        DC_y = self.CC(Y,35)
        return self.l1(DC_x,DC_y)



# loss_f = CC_loss()
# input_ = torch.ones((1,3,256,256))
# out = loss_f(input_,input_)