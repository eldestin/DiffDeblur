import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
# from models.Segmodels.segformer_mit import SegFormer
from collections import OrderedDict
from .model_utils import LayerNorm2d, DCNConv, DCNnorm, SimpleGate


     
class NAFBlockModified(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_heads = 1,deform = True, norm = True):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        if deform and norm:
            print("Channels: ", c)
            print("Group Number: ", c//16)
            self.conv2 = DCNnorm(channels= dw_channel, kernel_size=3, group = c//16, output_bias=True)
        elif deform and not norm:
            self.conv2 = DCNConv(channels= dw_channel, kernel_size=3, group = c//16, output_bias=True)
        else:
            self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # Deep feature transformation, utilize self attention to check the input details
        # self.sca = TransformerBlock(dim=dw_channel // 2, num_heads=num_heads, ffn_expansion_factor=FFN_Expand,bias=True,
        #                             LayerNorm_type = 'WithBias')
        # SimpleGate
        # activation
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNetModified(nn.Module):

    def __init__(self, img_channel=3, out_channel=3,width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], num_heads = [1,2,4,8],global_residual = False):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, (num, num_head) in enumerate(zip(enc_blk_nums, num_heads)):
            if i == 0:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlockModified(chan,num_heads=num_head,deform=True, norm=True) for _ in range(num)]
                    )
                )
                self.downs.append(
                    nn.Conv2d(chan, 2*chan, 2, 2)
                )
                chan = chan * 2
            elif i!= len(enc_blk_nums) -1 and i!=0:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlockModified(chan,num_heads=num_head,deform=True, norm=False) for _ in range(num)]
                    )
                )
                self.downs.append(
                    nn.Conv2d(chan, 2*chan, 2, 2)
                )
                chan = chan * 2
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlockModified(chan, num_heads=num_head, deform=False, norm = False) for _ in range(num)]
                    )
                )
                self.downs.append(
                    nn.Conv2d(chan, 2*chan, 2, 2)
                )
                chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlockModified(chan, num_heads = num_heads[-1],deform=False, norm = False) for _ in range(middle_blk_num)]
            )
        num_heads.reverse()
        for i, (num, num_head) in enumerate(zip(dec_blk_nums, num_heads)):
            if i==0:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
                chan = chan // 2
                self.decoders.append(
                    nn.Sequential(
                        *[NAFBlockModified(chan, num_heads=num_head,deform=False, norm = False) for _ in range(num)]
                    )
                )
            elif i == len(dec_blk_nums) - 1:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
                chan = chan // 2
                self.decoders.append(
                    nn.Sequential(
                        *[NAFBlockModified(chan, num_heads=num_head,deform=True, norm=True) for _ in range(num)]
                    )
                )
            else:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
                chan = chan // 2
                self.decoders.append(
                    nn.Sequential(
                        *[NAFBlockModified(chan, num_heads = num_head,deform=True, norm=False) for _ in range(num)]
                    )
                )
        self.padder_size = 2 ** len(self.encoders)
        self.global_residual = global_residual
    def convert_pl(self, path):
        '''
        This function aims to convert PT lightning parameters dictionary into torch load state dict 
        '''
        ckpt = torch.load(path,map_location='cpu')
        new_state_dict = OrderedDict()
        for k in ckpt['state_dict']:
            # print(k)
            #name = 1
            # print(k[:4])
            if k[:4] != 'net.':
            #if 'tiny_unet.' not in k:
                continue
            name = k.replace('net.','')
            # print(name)
            new_state_dict[name] = ckpt['state_dict'][k]
        return new_state_dict

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        # dpt_map = self.depth_model(inp)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        if self.global_residual:
            x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

