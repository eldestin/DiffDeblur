import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16, vgg19 
from torch.cuda import amp
class VGG_loss(nn.Module):
    '''
    This is the implementation of VGG Loss
        -> feature-based loss instead pixel-based loss to solve misalignment problem
    '''
    def __init__(self):
        super(VGG_loss, self).__init__()
        self.vgg_model = vgg19(pretrained=True).eval()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.Snd_net = nn.Sequential(*list(self.vgg_model.features)[:3]).eval()
        self.Svnth_net = nn.Sequential(*list(self.vgg_model.features)[:8]).eval()
        self.forteenth_net = nn.Sequential(*list(self.vgg_model.features)[:15]).eval()
    def forward(self,predict, clear):
        with amp.autocast():
            loss_1 = self.l1_loss(self.Snd_net(predict), self.Snd_net(clear))
            loss_2 = self.l1_loss(self.Svnth_net(predict), self.Svnth_net(clear))
            loss_3 = self.l1_loss(self.forteenth_net(predict), self.forteenth_net(clear))
            loss_all = 1/3*(loss_1 + loss_2 + loss_3)
        return loss_all

