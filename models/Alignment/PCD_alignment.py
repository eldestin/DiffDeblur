import torch
import sys
sys.path.append("..")
sys.path.append("../..")
from torch import nn
from torch.nn import functional as F
from DCNv4 import modules as opsm
from einops import rearrange
from models.Alignment.Alignformer import double_conv_down, single_conv, double_conv
# from Alignformer import double_conv_down, single_conv, double_conv

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
    
class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
    
class Pyramid_fea(nn.Module):
    '''
    The pyramid feature extraction module
    '''
    def __init__(self, in_channel = 6, num_features = 64):
        super(Pyramid_fea, self).__init__()
        self.in_channel = in_channel
        self.n_feat = num_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.n_feat, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True)
        )
        self.fea_extract = nn.Sequential(ResidualBlockNoBN(num_features),
                                         ResidualBlockNoBN(num_features))
        
        self.Downsample1 = double_conv_down(in_ch=self.n_feat, out_ch=self.n_feat)
        self.Downsample2 = double_conv_down(in_ch=self.n_feat, out_ch=self.n_feat)

    def forward(self, x):
        # original size
        x_in = self.conv1(x)
        x1 = self.fea_extract(x_in)
        # H/2
        x2 = self.Downsample1(x1)
        # H/4
        x3 = self.Downsample2(x2)
        return x1, x2, x3
    
class PCDAlignment(nn.Module):
    '''
    Here is the implementation for alignment module;
    Using: Pyramid, Cascading and deformable convolution (EDVR, ADNet)
    
    Params:
        1. num_features: channel number;
        2. deformable group: 
    '''
    def __init__(self, num_features, groups = 4, ):
        super(PCDAlignment, self).__init__()
        # Three levels 
        # Level 3, 1/4 spatial size
        self.L3_conv1 = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias = True)
        self.L3_conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias = True)
        self.L3_dcn = DCNConv(num_features, 3, groups,output_bias=True)

        # Level 2, 1/2 size
        self.L2_conv1 = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias = True)
        self.L2_conv2 = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias = True)
        self.L2_conv3 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias = True)
        self.L2_dcn = DCNConv(num_features, 3, groups,output_bias=True)
        self.L2_fea_conv = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias = True)

        # Original level
        self.L1_conv1 = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias = True)
        self.L1_conv2 = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias = True)
        self.L1_conv3 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias = True)
        self.L1_dcn = DCNConv(num_features, 3, groups,output_bias=True)
        self.L1_fea_conv = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias = True)

        # casading dcn
        self.cas_conv1 = nn.Conv2d(num_features * 2, num_features, 3, 1, 1, bias = True)
        self.cas_conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)
        self.cas_dcn = DCNConv(num_features, 3, groups,output_bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, l, r):
        '''
            l = [l1, l2, l3]
            r = [r1, r2, r3]
            each (b,c,h,w), (b,c,h/2,w/2), (b,c,h/4,w/4)
        '''
        l1, l2, l3 = l
        r1, r2, r3 = r
        # For L3
        level3_f = torch.cat([l3, r3], dim = 1)
        level3_f = self.lrelu(self.L3_conv1(level3_f))
        level3_f = self.lrelu(self.L3_conv2(level3_f))
        l3_fea = self.L3_dcn(level3_f)

        # For L2
        level2_f = torch.cat([l2, r2], dim = 1)
        level2_f = self.lrelu(self.L2_conv1(level2_f))
        # add interpolation
        level3_f = F.interpolate(level3_f, scale_factor=2, mode="bilinear", align_corners=False)
        level2_f = self.lrelu(self.L2_conv2(torch.cat([level2_f, level3_f * 2], dim = 1)))
        level2_f = self.lrelu(self.L2_conv3(level2_f))
        l2_fea = self.L2_dcn(level2_f)
        # aggregation
        l3_fea = F.interpolate(l3_fea, scale_factor=2, mode="bilinear", align_corners=False)
        l2_fea = self.lrelu(self.L2_fea_conv(torch.cat([l2_fea, l3_fea], dim = 1)))

        # For L1
        level1_f = torch.cat([l1, r1], dim = 1)
        level1_f = self.lrelu(self.L1_conv1(level1_f))
        # add interpolation
        level2_f = F.interpolate(level2_f, scale_factor=2, mode="bilinear", align_corners=False)
        level1_f = self.lrelu(self.L1_conv2(torch.cat([level1_f, level2_f * 2], dim = 1)))
        level1_f = self.lrelu(self.L1_conv3(level1_f))
        l1_fea = self.L1_dcn(level1_f)
        # aggregation
        l2_fea = F.interpolate(l2_fea, scale_factor=2, mode="bilinear", align_corners=False)
        l1_fea = self.lrelu(self.L1_fea_conv(torch.cat([l1_fea, l2_fea], dim = 1)))

        # casading
        fea = torch.cat([l1_fea, r1], dim = 1)
        fea = self.lrelu(self.cas_conv1(fea))
        fea = self.lrelu(self.cas_conv2(fea))
        l1_fea = self.cas_dcn(fea)

        return l1_fea


class Spatial_att(nn.Module):
    def __init__(self, num_fea, fea_out ,bias = False):
        super(Spatial_att, self).__init__()
        self.att1 = nn.Conv2d(num_fea * 2, num_fea * 2, kernel_size=3, padding=1, bias=bias)
        self.att2 = nn.Conv2d(num_fea * 2, fea_out, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
    def forward(self, l, r):
        '''
            l, r are concatenation of image with corresponding depth map
        '''
        fea_cat = torch.cat([l, r], dim = 1)
        att_map = self.relu(self.att2(self.att1(fea_cat)))
        return att_map

class Align_PCD(nn.Module):
    def __init__(self, num_channel, num_features, out_ch = 3):
        super(Align_PCD, self).__init__()
        self.pyramid = Pyramid_fea(num_channel, num_features=num_features)
        self.align_module = PCDAlignment(num_features=num_features)
        self.feature_extract = nn.Sequential(
            nn.Conv2d(num_channel, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.att = Spatial_att(num_features, num_features, bias=True)
        self.conv = nn.Sequential(single_conv(num_features, num_features), 
                        nn.Conv2d(num_features, out_ch, 3, 1, 1))
    def forward(self, l, r, dpt_map):
        l_dpt, r_dpt = dpt_map
        l, r = torch.cat([l, l_dpt], dim = 1), torch.cat([r, r_dpt], dim = 1)
        f_l = self.pyramid(l)
        f_r = self.pyramid(r)
        align_out = self.align_module(f_l, f_r)
        # att map
        fea_l = self.feature_extract(l)
        fea_r = self.feature_extract(r)
        att_map = self.att(fea_l, fea_r)
        # element wise multiplication
        out = self.conv(align_out * att_map)
        # print(align_out.shape)
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda")
    dpt_map = [torch.randn((2, 1, 192, 192)).to(device),torch.randn((2, 1, 192, 192)).to(device)]
    l, r = torch.randn((2, 3, 192, 192)).to(device),torch.randn((2, 3, 192, 192)).to(device)
    model = Align_PCD(4, 64).to(device)
    out = model(l, r, dpt_map)
    print(out.shape)


