import torch
import torch.nn as nn
import torch.nn.functional as F
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


#[x4_out,x8_out,x16_out]
class MultiScaleSupervisionLoss(nn.Module):
    def __init__(self):
        super(MultiScaleSupervisionLoss, self).__init__()
        self.CL1 = L1_Charbonnier_loss()
        self.weight = [0.5,0.5]#,0.25,0.25]
    def forward(self,preds,gt):
        loss_sum = 0
        losses = [self.weight[i]*self.CL1(pred,F.interpolate(gt,scale_factor = 0.5**(i+1))) for (i,pred) in enumerate(preds)]
        for loss in losses:
            loss_sum +=loss
        return loss_sum
    
