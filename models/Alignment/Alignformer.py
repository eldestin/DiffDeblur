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

class Cross_attention_dcn(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim ,num_heads, bias):
        super(Cross_attention_dcn, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.wq = nn.Conv2d(dim_q, dim, kernel_size=1, bias=bias)
        self.wk = nn.Conv2d(dim_k, dim, kernel_size=1, bias=bias)
        self.wv = nn.Conv2d(dim_v, dim, kernel_size=1, bias=bias)
        self.wq_dcn = DCNConv(dim, kernel_size=3,group=4, output_bias=bias)
        self.wk_dcn = DCNConv(dim, kernel_size=3,group=4, output_bias=bias)
        self.wv_dcn = DCNConv(dim, kernel_size=3,group=4, output_bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, q, k, v):
        b,c,h,w = q.shape

        q, k, v = self.wq_dcn(self.wq(q)), self.wk_dcn(self.wk(k)), self.wv_dcn(self.wv(v))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # print("Rearrange q shape: ", q.shape)
        # print("Rearrange k shape: ", k.shape)
        # print("Rearrange v shape: ", v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class Cross_attention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim ,num_heads, bias):
        super(Cross_attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.wq = nn.Conv2d(dim_q, dim, kernel_size=1, bias=bias)
        # self.wkv = nn.Conv2d(dim_k, dim * 2, kernel_size=1, bias=bias)
        self.wk = nn.Conv2d(dim_k, dim, kernel_size=1, bias=bias)
        self.wv = nn.Conv2d(dim_v, dim, kernel_size=1, bias=bias)
        # self.w_q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        # self.w_kv = nn.Conv2d(dim * 2, dim * 2, kernel_size= 3, stride=1, padding=1, groups=dim*2,bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, q, k, v):
        b,c,h,w = q.shape

        # q, k, v = self.w_share(self.wq(q)), self.w_share(self.wk(k)), self.w_share(self.wv(v))
        q, k, v = self.wq(q), self.wk(k), self.wv(v)
        # kv = self.w_kv(kv)
        # k, v = kv.chunk(2, dim = 1)
        # print(q.shape)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # print("Rearrange q shape: ", q.shape)
        # print("Rearrange k shape: ", k.shape)
        # print("Rearrange v shape: ", v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Self_attention(nn.Module):
    '''
    The implementation of self attention;
    '''
    def __init__(self, dim, dim_out, num_heads, bias):
        super(Self_attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim_out, kernel_size=1, bias=bias)
        
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
    

class Self_attention_dcn(nn.Module):
    '''
    The implementation of self attention;
    '''
    def __init__(self, dim, dim_out, num_heads, bias):
        super(Self_attention_dcn, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = DCNConv(channels=dim*3, kernel_size=3, dw_kernel_size=3,
                                  stride=1, pad=1, group=4)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim_out, kernel_size=1, bias=bias)
        
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
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),)

    def forward(self, x):
        return self.conv(x)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class double_conv_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class double_conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_up, self).__init__()
        self.conv = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                  nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))

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

class PosEnSine(nn.Module):
    """
    Code borrowed from DETR: models/positional_encoding.py
    output size: b*(2.num_pos_feats)*h*w
    """

    def __init__(self, num_pos_feats):
        super(PosEnSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = True
        self.scale = 2 * math.pi
        self.temperature = 10000

    def forward(self, x):
        b, c, h, w = x.shape
        not_mask = torch.ones(1, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.repeat(b, 1, 1, 1)
        return pos

class MLP(nn.Module):
    """
    conv-based MLP layers.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.linear1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class Transformer_enc_self(nn.Module):
    def __init__(self, feature_dims, n_head = 4, mlp_ratio = 2, depth_cat = False):
        super(Transformer_enc_self, self).__init__()
        dim_q, dim_k, dim_v, feature_dim = feature_dims
        # self.pos_enc = PosEnSine(feature_dim)
        # if depth_cat:
        #     self.pos_enc = PosEnSine(feature_dim)
        self.att = Self_attention_dcn(feature_dim * 2, feature_dim ,n_head, False)
        mlp_hidden = int(feature_dim * mlp_ratio)
        self.FFN = MLP(feature_dim, mlp_hidden)
        self.norm = nn.GroupNorm(1,  feature_dim)
        self.res = double_conv(dim_q, feature_dim)
    def forward(self, qkv):
        # qkv_pos = self.pos_enc(qkv)
        # print("QKV shape: ", qkv.shape)
        att_out = self.att(qkv)
        # print("Attention output shape: ", att_out.shape)
        first_res = self.res(qkv) + att_out
        # pass MLP
        mlp_out = self.FFN(first_res)
        second_res = mlp_out + first_res
        out = self.norm(second_res)
        return out
    
class Transformer_enc(nn.Module):
    def __init__(self, feature_dims, n_head = 4, mlp_ratio = 2, depth_cat = False):
        super(Transformer_enc, self).__init__()
        dim_q, dim_k, dim_v, feature_dim = feature_dims
        self.pos_enc = PosEnSine(feature_dim//2)
        if depth_cat:
            self.pos_enc = PosEnSine(feature_dim)
        self.att = Cross_attention(dim_q, dim_k, dim_v, feature_dim, n_head, False)
        mlp_hidden = int(feature_dim * mlp_ratio)
        self.FFN = MLP(feature_dim, mlp_hidden)
        self.norm = nn.GroupNorm(1,  feature_dim)
        self.res = double_conv(dim_q, feature_dim)
    def forward(self, q, k, v):
        q_pos = self.pos_enc(q)
        k_pos = self.pos_enc(k)
        # for test only
        #k_pos = torch.cat([k_pos, k_pos, k_pos], dim = 1)
        # print(q.shape, q_pos.shape)
        # print(k_pos.shape)
        # print(k.shape)
        att_out = self.att(q + q_pos, k + k_pos, v)
        # att_out = self.att(q , k , v)
        # print("Attention output shape: ", att_out.shape)
        first_res = self.res(q) + att_out
        # pass MLP
        mlp_out = self.FFN(first_res)
        second_res = mlp_out + first_res
        out = self.norm(second_res)
        return out

    
class Transformer_enc_dcn(nn.Module):
    def __init__(self, feature_dims, n_head = 4, mlp_ratio = 2, depth_cat = False):
        super(Transformer_enc_dcn, self).__init__()
        dim_q, dim_k, dim_v, feature_dim = feature_dims
        self.pos_enc = PosEnSine(feature_dim//2)
        if depth_cat:
            self.pos_enc = PosEnSine(feature_dim)
        self.att = Cross_attention_dcn(dim_q, dim_k, dim_v, feature_dim, n_head, False)
        mlp_hidden = int(feature_dim * mlp_ratio)
        self.FFN = MLP(feature_dim, mlp_hidden)
        self.norm = nn.GroupNorm(1,  feature_dim)
        self.res = double_conv(dim_q, feature_dim)
    def forward(self, q, k, v):
        q_pos = self.pos_enc(q)
        k_pos = self.pos_enc(k)
        att_out = self.att(q + q_pos, k + k_pos, v)
        # print("Attention output shape: ", att_out.shape)
        first_res = self.res(q) + att_out
        # pass MLP
        mlp_out = self.FFN(first_res)
        second_res = mlp_out + first_res
        out = self.norm(second_res)
        return out

class Alignformer_self(nn.Module):
    '''
       The implementation of utilizing depth map to guide attention (concat/cross attention)
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False, module = "cat"):
        super(Alignformer_self, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        self.DAM = DAModule(in_ch = src_channel, feat_ch = feature_dim, out_ch = src_channel,
                            demodulate = True, requires_grad = True)
        # Define feature extractor
        # 孪生
        self.unet_q = Unet(src_channel, feature_dim, feature_dim)
        self.unet_kv = Unet(src_channel, feature_dim, feature_dim)
        self.change_dim = nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1, bias=False)
        self.depth_cdim = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope= 0.2, inplace=True)
        self.unet_d = Unet(1, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Transformer_enc_self(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Transformer_enc_self(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Transformer_enc_self(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
        ])
        self.guide_module = module
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        src = self.DAM(x_l, x_r)
        q_fea, l_fea, r_fea = self.unet_q(src), self.unet_kv(x_l), self.unet_kv(x_r)
        depth_feature_l = self.unet_d(l_dpt)
        depth_feature_r = self.unet_d(r_dpt)
        outputs = []
        if self.guide_module == "cat":
            for i in range(3):
                # print("Query feature shape:",q_feature[i+3].shape)
                # print("Depth feature shape:",depth_feature_l[i+3].shape)
                # print("Key feature shape:",k_feature[i+3].shape)
                qkv = self.change_dim(torch.cat([q_fea[i+3], l_fea[i+3], r_fea[i+3]], dim = 1))
                qkv_depth = self.depth_cdim(torch.cat([depth_feature_l[i+3], depth_feature_r[i+3]], dim = 1))
                outputs.append(self.trans_unit[i](
                    torch.cat([qkv, qkv_depth], dim = 1)
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


class Alignformer_self_noDAM(nn.Module):
    '''
       The implementation of utilizing depth map to guide attention (concat/cross attention)
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False, module = "cat"):
        super(Alignformer_self_noDAM, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        # Define feature extractor
        self.unet_qkv = Unet(src_channel * 2, feature_dim, feature_dim)
        self.unet_d = Unet(1 * 2, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Transformer_enc_self(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Transformer_enc_self(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Transformer_enc_self(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
        ])
        self.guide_module = module
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        qkv_feature = self.unet_qkv(torch.cat([x_l, x_r], dim = 1))
        depth_feature = self.unet_d(torch.cat([l_dpt, r_dpt], dim = 1))
        outputs = []
        if self.guide_module == "cat":
            for i in range(3):
                # print("Query feature shape:",q_feature[i+3].shape)
                # print("Depth feature shape:",depth_feature_l[i+3].shape)
                # print("Key feature shape:",k_feature[i+3].shape)
                outputs.append(self.trans_unit[i](
                    torch.cat([qkv_feature[i+3], depth_feature[i+3]], dim = 1)
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


class Alignformer_self_predict(nn.Module):
    '''
       The implementation without depth map -> more like noise 
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False, module = "cat"):
        super(Alignformer_shallow_guided, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        # Define feature extractor
        # utilize siamese network, one network for both l and r image
        # Unet feature extracted
        self.unet_shared = Unet(src_channel, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Transformer_enc_self(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Transformer_enc_self(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Transformer_enc_self(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
        ])
        self.guide_module = module
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        l_fea, r_fea = self.unet_l(torch.cat([x_l, l_dpt], dim = 1)), self.unet_r(torch.cat([x_r, r_dpt], dim=1))
        outputs = []
        if self.guide_module == "cat":
            for i in range(3):
                # print("Query feature shape:",q_feature[i+3].shape)
                # print("Depth feature shape:",depth_feature_l[i+3].shape)
                # print("Key feature shape:",k_feature[i+3].shape)
                outputs.append(self.trans_unit[i](
                    torch.cat([l_fea[i+3], r_fea[i+3]], dim = 1)
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

class Alignformer_shallow_guided(nn.Module):
    '''
       The implementation of utilizing depth map to guide attention (concat/cross attention)
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False, module = "cat"):
        super(Alignformer_shallow_guided, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        # Define feature extractor
        self.unet_l = Unet(src_channel + 1, feature_dim, feature_dim)
        self.unet_r = Unet(src_channel + 1, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Transformer_enc_self(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Transformer_enc_self(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Transformer_enc_self(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
        ])
        self.guide_module = module
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        l_fea, r_fea = self.unet_l(torch.cat([x_l, l_dpt], dim = 1)), self.unet_r(torch.cat([x_r, r_dpt], dim=1))
        outputs = []
        if self.guide_module == "cat":
            for i in range(3):
                # print("Query feature shape:",q_feature[i+3].shape)
                # print("Depth feature shape:",depth_feature_l[i+3].shape)
                # print("Key feature shape:",k_feature[i+3].shape)
                outputs.append(self.trans_unit[i](
                    torch.cat([l_fea[i+3], r_fea[i+3]], dim = 1)
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

class Alignformer_depMapG_dcn(nn.Module):
    '''
       The implementation of utilizing depth map to guide attention (concat/cross attention)
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False, module = "cat"):
        super(Alignformer_depMapG_dcn, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        self.DAM = DAModule(in_ch = src_channel, feat_ch = feature_dim, out_ch = src_channel,
                            demodulate = True, requires_grad = True)
        # Define feature extractor
        self.unet_q = Unet(src_channel, feature_dim, feature_dim)
        self.unet_k = Unet(src_channel, feature_dim, feature_dim)
        self.unet_d = Unet(1, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Transformer_enc_dcn(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Transformer_enc_dcn(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Transformer_enc_dcn(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
        ])
        self.guide_module = module
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        src = self.DAM(x_l, x_r)
        q_feature = self.unet_q(x_l)
        k_feature = self.unet_k(x_r)
        depth_feature_l, depth_feature_r = self.unet_d(l_dpt), self.unet_d(r_dpt)
        outputs = []
        if self.guide_module == "cat":
            for i in range(3):
                # print("Query feature shape:",q_feature[i+3].shape)
                # print("Depth feature shape:",depth_feature_l[i+3].shape)
                # print("Key feature shape:",k_feature[i+3].shape)
                outputs.append(self.trans_unit[i](
                    torch.cat([q_feature[i+3], depth_feature_l[i+3]], dim = 1), 
                    torch.cat([k_feature[i+3], depth_feature_r[i+3]], dim = 1), k_feature[i+3]
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
    

class Alignformer(nn.Module):
    '''
       The implementation of utilizing depth map to guide attention (concat/cross attention)
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False):
        super(Alignformer, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        self.DAM = DAModule(in_ch = src_channel, feat_ch = feature_dim, out_ch = src_channel,
                            demodulate = True, requires_grad = True)
        # Define feature extractor
        self.unet_q = Unet(src_channel, feature_dim, feature_dim)
        self.unet_k = Unet(src_channel, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Transformer_enc(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Transformer_enc(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Transformer_enc(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
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
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        x_l, x_r = torch.cat([x_l, l_dpt], dim = 1), torch.cat([x_r, r_dpt], dim = 1)
        src = self.DAM(x_l, x_r)
        q_feature = self.unet_q(x_l)
        k_feature = self.unet_k(x_r)
        outputs = []
    
        for i in range(3):
            # print("Query feature shape:",q_feature[i+3].shape)
            # print("Depth feature shape:",depth_feature_l[i+3].shape)
            # print("Key feature shape:",k_feature[i+3].shape)
            outputs.append(self.trans_unit[i](
                q_feature[i+3], 
                k_feature[i+3], k_feature[i+3]
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
    
    
class Alignformer_depMapG(nn.Module):
    '''
       The implementation of utilizing depth map to guide attention (concat/cross attention)
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False, module = "cat"):
        super(Alignformer_depMapG, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        self.DAM = DAModule(in_ch = src_channel, feat_ch = feature_dim, out_ch = src_channel,
                            demodulate = True, requires_grad = True)
        # Define feature extractor
        self.unet_q = Unet(src_channel, feature_dim, feature_dim)
        self.unet_k = Unet(src_channel, feature_dim, feature_dim)
        self.unet_d = Unet(1, feature_dim, feature_dim)
        # self.unet_qkv = Unet(src_channel * 3, feature_dim, feature_dim)
        # self.unet_kv = Unet(src_channel * 2, feature_dim, feature_dim)
        # self.unet_d = Unet(1 * 2, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Transformer_enc(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Transformer_enc(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Transformer_enc(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
        ])
        self.guide_module = module
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        # qkv_feature = self.unet_qkv(torch.cat([src, x_l, x_r], dim = 1))
        # depth_feature= self.unet_d(torch.cat([l_dpt, r_dpt], dim = 1))
        src = self.DAM(x_l, x_r)
        q_feature = self.unet_q(x_l)
        k_feature = self.unet_k(x_r)
        depth_feature_l, depth_feature_r = self.unet_d(l_dpt), self.unet_d(r_dpt)
        outputs = []
        if self.guide_module == "cat":
            for i in range(3):
                # print("Query feature shape:",q_feature[i+3].shape)
                # print("Depth feature shape:",depth_feature_l[i+3].shape)
                # print("Key feature shape:",k_feature[i+3].shape)
                outputs.append(self.trans_unit[i](
                    # torch.cat([q_feature[i+3], depth_feature_l[i+3]], dim = 1),
                    torch.cat([q_feature[i+3], depth_feature_l[i+3]], dim = 1), 
                    torch.cat([k_feature[i+3], depth_feature_r[i+3]], dim = 1),
                    k_feature[i+3]
                    # torch.cat([k_feature[i+3], depth_feature_r[i+3]], dim = 1)
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


class Transformer_enc_guide(nn.Module):
    '''
        The implementation of guidance utilizing depth map (cross attention)
    '''
    def __init__(self, feature_dims, n_head = 4, mlp_ratio = 2, depth_cat = False):
        super(Transformer_enc_guide, self).__init__()
        self.tf_block_l = Transformer_enc(feature_dims, n_head, mlp_ratio, depth_cat)
        self.tf_block_r = Transformer_enc(feature_dims, n_head, mlp_ratio, depth_cat)
        self.tf_block_all = Transformer_enc(feature_dims, n_head, mlp_ratio, depth_cat)

    def forward(self, x_l, x_r, dpt_map):
        # block_1 
        dpt_l, dpt_r = dpt_map
        att_l_out = self.tf_block_l(dpt_l, x_l, x_l)
        att_r_out = self.tf_block_r(dpt_r, x_r, x_r)
        att_out = self.tf_block_all(att_l_out, att_r_out, att_r_out)
        return att_out


class Alignformer_depMapCross(nn.Module):
    '''
        The implementation of utilizing depth map to guide attention (concat/cross attention)
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False, module = "cat"):
        super(Alignformer_depMapCross, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        self.DAM = DAModule(in_ch = src_channel, feat_ch = feature_dim, out_ch = src_channel,
                            demodulate = True, requires_grad = True)
        # Define feature extractor
        self.unet_q = Unet(src_channel, feature_dim, feature_dim)
        self.unet_k = Unet(src_channel, feature_dim, feature_dim)
        self.unet_d = Unet(1, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Transformer_enc_guide(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Transformer_enc_guide(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Transformer_enc_guide(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
        ])
        self.guide_module = module
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        src = self.DAM(x_l, x_r)
        q_feature = self.unet_q(x_l)
        k_feature = self.unet_k(x_r)
        depth_feature_l, depth_feature_r = self.unet_d(l_dpt), self.unet_d(r_dpt)
        outputs = []
        for i in range(3):
            # print("Query feature shape:",q_feature[i+3].shape)
            # print("Depth feature shape:",depth_feature_l[i+3].shape)
            # print("Key feature shape:",k_feature[i+3].shape)
            outputs.append(self.trans_unit[i](
                q_feature[i+3], 
                k_feature[i+3],
                [depth_feature_l[i+3], depth_feature_r[i+3]]
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
    

class Co_att_block(nn.Module):
    '''
    The implemntation of co-attention block;
    '''
    def __init__(self, feature_dims, n_head = 4, mlp_ratio = 2, depth_cat = False):
        super(Co_att_block, self).__init__()
        self.coatt1 = Transformer_enc(feature_dims=feature_dims, n_head=n_head, mlp_ratio=mlp_ratio, depth_cat=depth_cat)
        self.coatt2 = Transformer_enc(feature_dims=feature_dims, n_head=n_head, mlp_ratio=mlp_ratio, depth_cat=depth_cat)
        self.mlp_fusion = MLP(in_features=feature_dims[-1] * 2, out_features=feature_dims[-1])
    
    def forward(self, l_q, l_v, r_q, r_v):
        l_f = self.coatt1(l_q, r_q, r_v)
        r_f = self.coatt2(r_q, l_q, l_v)
        out = self.mlp_fusion(torch.cat([l_f, r_f], dim = 1))
        return out
    
class Alignformer_coatt(nn.Module):
    '''
       The implementation of utilizing depth map to guide attention (concat/cross attention)
    '''
    def __init__(self, feature_dims, src_channel, ref_channel, 
                    out_channel, n_head = 4, mlp_ratio = 2, depth_cat = False, module = "cat"):
        super(Alignformer_coatt, self).__init__()
        feature_dim = feature_dims[-1]
        # print(feature_dim)
        # self.DAM = DAModule(in_ch = src_channel, feat_ch = feature_dim, out_ch = src_channel,
        #                     demodulate = True, requires_grad = True)
        # Define feature extractor
        self.unet_q = Unet(src_channel, feature_dim, feature_dim)
        self.unet_k = Unet(src_channel, feature_dim, feature_dim)
        self.unet_d = Unet(1, feature_dim, feature_dim)
        # Define GAM
        self.trans_unit = nn.ModuleList([
            Co_att_block(feature_dims, n_head, mlp_ratio,depth_cat = depth_cat),
            Co_att_block(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat),
            Co_att_block(feature_dims, n_head, mlp_ratio, depth_cat = depth_cat)
        ])
        self.guide_module = module
        # Unet result output
        self.conv0 = double_conv(feature_dim, feature_dim)
        self.conv1 = double_conv_down(feature_dim, feature_dim)
        self.conv2 = double_conv_down(feature_dim, feature_dim)
        self.conv3 = double_conv(feature_dim, feature_dim)
        self.conv4 = double_conv_up(feature_dim, feature_dim)
        self.conv5 = double_conv_up(feature_dim, feature_dim)
        self.conv6 = nn.Sequential(single_conv(feature_dim, feature_dim), 
                        nn.Conv2d(feature_dim, out_channel, 3, 1, 1))
        
    def forward(self, x_l, x_r, dpt_map):
        l_dpt, r_dpt = dpt_map
        q_feature = self.unet_q(x_l)
        k_feature = self.unet_k(x_r)
        depth_feature_l, depth_feature_r = self.unet_d(l_dpt), self.unet_d(r_dpt)
        outputs = []
        if self.guide_module == "cat":
            for i in range(3):
                # print("Query feature shape:",q_feature[i+3].shape)
                # print("Depth feature shape:",depth_feature_l[i+3].shape)
                # print("Key feature shape:",k_feature[i+3].shape)
                outputs.append(self.trans_unit[i](
                    # torch.cat([q_feature[i+3], depth_feature_l[i+3]], dim = 1),
                    torch.cat([q_feature[i+3], depth_feature_l[i+3]], dim = 1), 
                    q_feature[i+3],
                    torch.cat([k_feature[i+3], depth_feature_r[i+3]], dim = 1),
                    k_feature[i+3]
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
    

