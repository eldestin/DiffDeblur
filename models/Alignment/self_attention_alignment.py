import sys
sys.path.append("../..")
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
# from basicsr.archs.pwcnet_arch import FlowGenerator
from einops import rearrange
from models.Alignment.DAM import DAModule
# from DAM import DAModule
import math
import numbers
from DCNv4 import modules as opsm

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Self_attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    

class Self_attention(nn.Module):
    '''
    The implementation of self attention;
    '''
    def __init__(self, dim, num_heads, bias):
        super(Self_attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        # qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)  
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=True),)

    def forward(self, x):
        return self.conv(x)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)


class double_conv_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)


class double_conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_up, self).__init__()
        self.conv = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                  nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_ch, feat_ch, out_ch):
        super().__init__()
        self.conv_in = single_conv(in_ch, feat_ch)

        self.conv1 = double_conv_down(feat_ch, feat_ch)
        self.conv2 = double_conv_down(feat_ch, feat_ch)
        self.conv3 = double_conv(feat_ch, feat_ch)
        self.conv4 = double_conv_up(feat_ch, feat_ch)
        self.conv5 = double_conv_up(feat_ch, feat_ch)
        self.conv6 = double_conv(feat_ch, out_ch)

    def forward(self, x):
        feat0 = self.conv_in(x)    # H, W
        feat1 = self.conv1(feat0)   # H/2, W/2
        feat2 = self.conv2(feat1)    # H/4, W/4
        feat3 = self.conv3(feat2)    # H/4, W/4
        feat3 = feat3 + feat2     # H/4
        feat4 = self.conv4(feat3)    # H/2, W/2
        feat4 = feat4 + feat1    # H/2, W/2
        feat5 = self.conv5(feat4)   # H
        feat5 = feat5 + feat0   # H
        feat6 = self.conv6(feat5)

        return feat0, feat1, feat2, feat3, feat4, feat6


class Alignformer_self_predict(nn.Module):
    '''
       The implementation without depth map -> more like noise 
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, ffn_expansion_factor = 2, bias = True):
        super(Alignformer_self_predict, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        # Define feature extractor
        # utilize siamese network, one network for both l and r image
        # Unet feature extracted
        self.unet_shared = Unet(src_channel, feature_dim, feature_dim)
        # single conv 1*1
        self.change_dim = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=bias)
        self.act = nn.LeakyReLU(negative_slope= 0.2, inplace=True)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            TransformerBlock(feature_dim, n_head, ffn_expansion_factor = ffn_expansion_factor,bias=bias, LayerNorm_type="BiasFree"),
            TransformerBlock(feature_dim, n_head, ffn_expansion_factor = ffn_expansion_factor,bias=bias, LayerNorm_type="BiasFree"),
            TransformerBlock(feature_dim, n_head, ffn_expansion_factor = ffn_expansion_factor,bias=bias, LayerNorm_type="BiasFree")
        ])
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r):
        l_fea, r_fea = self.unet_shared(x_l), self.unet_shared(x_r)

        outputs = []
        for i in range(3):
            # print("Query feature shape:",q_feature[i+3].shape)
            # print("Depth feature shape:",depth_feature_l[i+3].shape)
            # print("Key feature shape:",k_feature[i+3].shape)
            qkv = self.act(self.change_dim(torch.cat([l_fea[i+3], r_fea[i+3]], dim = 1)))
            outputs.append(self.trans_unit[i](
                qkv
            ))
        f0 = self.conv0(outputs[2])  # H, W
        f1 = self.conv1(f0)  # H/2, W/2
        f1 = f1 + outputs[1]
        f2 = self.conv2(f1)  # H/4, W/4
        f2 = f2 + outputs[0]
        f3 = self.conv3(f2)  # H/4, W/4
        f3 = f3 + outputs[0] + f2
        f4 = self.conv4(f3)   # H/2, W/2
        f4 = f4 + outputs[1] + f1
        f5 = self.conv5(f4)   # H, W
        f5 = f5 + outputs[2] + f0
        out = self.conv6(f5)

        return out