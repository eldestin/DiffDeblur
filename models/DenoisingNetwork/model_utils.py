import torch
from torch import nn
import sys
from einops import rearrange, reduce
from DCNv4 import modules as opsm
sys.path.append("../..")
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g
    
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    

deform_conv=getattr(opsm, 'DCNv4')
class DCNConv(nn.Module):
    """
    DCNv4
    """
    def __init__(self,channels,kernel_size,group,offset_scale=1.0, output_bias=False):
        super(DCNConv, self).__init__()

        self.dcn_conv = deform_conv(
                channels=channels,
                group=group,
                offset_scale=offset_scale,
                dw_kernel_size=kernel_size,
                output_bias=output_bias
            )
    
    def forward(self, inp):
        _,_,H,W = inp.size()
        inp = rearrange(inp, 'b c h w -> b (h w) c').contiguous()
        out = self.dcn_conv(inp,shape=[H,W])
        out = rearrange(out, 'b (h w) c -> b c h w',h=H,w=W).contiguous()
        return out
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class DCNnorm(nn.Module):
    '''
    Utilize the normalization after dcn to ensure the training stability;
    '''
    def __init__(self,channels,kernel_size,group,offset_scale=1.0, output_bias=False):
        super().__init__()
        self.dcnconv = DCNConv(channels, kernel_size, group, offset_scale, output_bias)
        self.norm = LayerNorm2d(channels)
    
    def forward(self, x):
        return self.norm(self.dcnconv(x))