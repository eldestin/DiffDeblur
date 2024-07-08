import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torchvision import models,transforms
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
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
    
class resnet50(nn.Module):
    def __init__(self,pretrained=True):
        super(resnet50, self).__init__()
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        
        model = models.resnet50(pretrained=pretrained)
        self.head= nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        self.layers = nn.ModuleList([
            model.layer1,
            model.layer2,
            model.layer3]
        #    model.layer4]
        )

    def forward(self,x):
        fts=[]
        x = self.normalize(x)
        out = self.head(x)
        fts.append(out)
        for layer in self.layers:
            out = layer(out)
            fts.append(out)
        return fts

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g
    

class NAFAlignBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # # Siamese with concatenation as fusion -> RGM align
        self.siamese = nn.Sequential(LayerNorm(c),
                                     nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                                    #  nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,bias=True)
                                    DCNConv(channels=dw_channel, kernel_size=3, group=4, output_bias=True)
                                     )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c * 2, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm2 = LayerNorm(2 * c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x_l, x_r = inp

        l_f = self.sg(self.siamese(x_l))
        r_f = self.sg(self.siamese(x_r))
        x = torch.cat([l_f, r_f], dim = 1)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y_l = x_l + x * self.beta
        y_r = x_r + x * self.beta

        x = self.conv4(self.norm2(torch.cat([y_l, y_r], dim = 1)))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y_l + x * self.gamma, y_r + x * self.gamma
    
class NAFNetBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel//2, out_channels=dw_channel//2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # # Siamese with concatenation as fusion -> RGM align
        self.siamese = nn.Sequential(LayerNorm(c),
                                     nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                                    #  nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,bias=True)
                                    DCNConv(channels=dw_channel, kernel_size=3, group=4, output_bias=True)
                                     )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):

        # return y + x * self.gamma
        x = inp
        x = self.sg(self.siamese(x))
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNetAlign(nn.Module):

    def __init__(self, img_channel=3,out_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],global_residual=False):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width * 2, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        NAFBlock = NAFAlignBlock
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.out_Tanh = nn.Tanh()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.global_residual = global_residual

    def forward(self, inp):
        l,r = inp
        B, C, H, W = l.shape
        x_l, x_r = self.check_image_size(l),self.check_image_size(r)

        x_l = self.intro(l)
        x_r = self.intro(r)
        x = [x_l, x_r]
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            # each x become: x_l, x_r; each shape (B,C,H,W)
            x = encoder(x)
            encs.append(x)
            x_l, x_r = down(x[0]), down(x[1])
            x = [x_l, x_r]
        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x_l, x_r = up(x[0]), up(x[1])
            x_l, x_r = x_l + enc_skip[0], x_r + enc_skip[1]
            x = [x_l, x_r]
            x = decoder(x)
        
        x = self.ending(torch.cat(x, dim=1))
        if self.global_residual:
            print("In residual: ")
            print(x.shape)
            print(inp.shape)
            x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNet(nn.Module):

    def __init__(self, img_channel=3,out_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],global_residual=False):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        NAFBlock = NAFNetBlock
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.out_Tanh = nn.Tanh()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.global_residual = global_residual

    def forward(self, inp):
        inp = self.check_image_size(inp)
        B, C, H, W = inp.shape

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            # each x become: x_l, x_r; each shape (B,C,H,W)
            x = encoder(x)
            encs.append(x)
            x = down(x)
            # x = [x_l, x_r]
        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            # x = [x_l, x_r]
            x = decoder(x)
        
        x = self.ending(x)
        if self.global_residual:
            print("In residual: ")
            print(x.shape)
            print(inp.shape)
            x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == "__main__":
    l, r = torch.randn((2, 3, 64, 64)), torch.randn((2, 3, 64, 64))
    net = NAFNetAlign(3,3,64,1,[1,1,4],[1,1,1])
    # semantic = resnet50()
    # guide = semantic(l - r)
    # timesteps = torch.randint(0, 1000, (2,)).long()
    print(net([l,r]).shape)
