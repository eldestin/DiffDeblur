import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from .model_utils import LayerNorm, DCNConv, DCNnorm, SimpleGate
from torchvision import models, transforms
from diffusers.models.embeddings import Timesteps

class resnet18(nn.Module):
    def __init__(self,pretrained=True):
        super(resnet18, self).__init__()
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        
        model = models.resnet18(pretrained=pretrained)
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
    



class NAFBlockFusion(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., res_50 = False, dual = True, deform = False, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        if not deform:
            self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        elif deform and norm:
            # modified to dcn
            # only for larger feature map 
            print("Channels: ", c)
            print("Groups number: ", c//16)
            self.conv2 = DCNnorm(channels=dw_channel, kernel_size=3, group=c//16, output_bias=True)
        else:
            self.conv2 = DCNConv(channels=dw_channel, kernel_size=3, group=c//16, output_bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dual = dual
        if res_50 and dual:
            self.mlp_guided= nn.Sequential(
                # resnet 50
                # *2 为dual pixel 的情况，测试single iamge记得改回来
                nn.Conv2d(in_channels=64+256+512+1024,out_channels=c,kernel_size=1),
                # nn.Conv2d(in_channels=256+128+64+64,out_channels=c,kernel_size=1),
                nn.LeakyReLU()
            )
        elif res_50 and not dual:
            self.mlp_guided = nn.Sequential(
                # resnet 50
                # *2 为dual pixel 的情况，测试single iamge记得改回来
                nn.Conv2d(in_channels=64+256+512+1024,out_channels=c,kernel_size=1),
                # nn.Conv2d(in_channels=256+128+64+64,out_channels=c,kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=c,out_channels=c,kernel_size=1)
            )
        else: 
            self.mlp_guided = nn.Sequential(
                # resnet 50
                # nn.Conv2d(in_channels=64+256+512+1024,out_channels=c,kernel_size=1),
                nn.Conv2d(in_channels=256+128+64+64,out_channels=c,kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=c,out_channels=c,kernel_size=1)
            )
        self.mlp_fusion = nn.Conv2d(
            in_channels=c*2,
            out_channels=c,
            kernel_size=1
        )
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        # (b,c/4,1,1)  *  4
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
  
        inp, time, guided_fts = x
    
          
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp
        if guided_fts is not None:
           # print("In guided fst: ", guided_fts.shape)
            guided_fts_ = self.mlp_guided(guided_fts)
            x = self.mlp_fusion(torch.cat([guided_fts_,x],dim=1))

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x).contiguous()
        # with torch.no_grad():
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        # guide again follows Multi scale feature guidance 
        # The same feature guidance per each block
        if guided_fts is not None:
            x = self.mlp_fusion(torch.cat([guided_fts_,x], dim = 1))
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn

        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x) 

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time, guided_fts
    
class NAFNetFusionModifiedcat(nn.Module):

    def __init__(self, img_channel=3,out_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1,DW_Expand=2,Finetune=True, res_50=False, dual = True):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        #time_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_pos_emb = Timesteps(fourier_dim,flip_sin_to_cos=True,downscale_freq_shift=0.) 
        time_dim = width * 4
        self.finetune = Finetune
        if self.finetune:
            basic_block = NAFBlockFusion
        else:
            basic_block = NAFBlockFusion
            
        self.time_mlp = nn.Sequential(
            time_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )
        self.dual = dual
        # self.pre_seg = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)
        # self.pre_seg_in = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, padding=0, stride=1, groups=1,
        #                       bias=True)
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
        for i, num in enumerate(enc_blk_nums):
            if i == 1 or i == 2:
                self.encoders.append(
                    nn.Sequential(
                        *[basic_block(chan, time_dim,DW_Expand, res_50 = res_50, dual = dual, deform=True, norm = True) for _ in range(num)]
                    )
                )
                self.downs.append(
                    nn.Conv2d(chan, 2*chan, 2, 2)
                )
                chan = chan * 2
            elif i != len(enc_blk_nums) - 1:
                self.encoders.append(
                    nn.Sequential(
                        *[basic_block(chan, time_dim,DW_Expand, res_50 = res_50, dual = dual, deform=True, norm = False) for _ in range(num)]
                    )
                )
                self.downs.append(
                    nn.Conv2d(chan, 2*chan, 2, 2)
                )
                chan = chan * 2
            else:
                print("Block "+str(i)+" traditional conv initialized.")
                self.encoders.append(
                    nn.Sequential(
                        *[basic_block(chan, time_dim,DW_Expand, res_50 = res_50, dual = dual, deform=False, norm = False) for _ in range(num)]
                    )
                )
                self.downs.append(
                    nn.Conv2d(chan, 2*chan, 2, 2)
                )
                chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[basic_block(chan, time_dim, res_50 = res_50, dual = dual, deform = False, norm=False) for _ in range(middle_blk_num)]
            )

        for i, num in enumerate(dec_blk_nums):
            if i == 0:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
                chan = chan // 2
                self.decoders.append(
                    nn.Sequential(
                        *[basic_block(chan, time_dim, res_50 = res_50, dual = dual, deform=False, norm=False) for _ in range(num)]
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
                        *[basic_block(chan, time_dim, res_50 = res_50, dual = dual, deform=True, norm=True) for _ in range(num)]
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
                        *[basic_block(chan, time_dim, res_50 = res_50, dual = dual, deform=True, norm = False) for _ in range(num)]
                    )
                )

        self.padder_size = 2 ** (len(self.encoders))
        
        self.pre_net = None
        
    def ResizeGuided_fts(self,guided_fts,x):
        if guided_fts is None:
            return None
        B,C,H,W = x.size()
        guided_fts_resized = []
        for ft in guided_fts:
            guided_fts_resized.append(
                F.interpolate(ft,size=(H,W))
            )
        return torch.cat(guided_fts_resized,dim=1)

    def ResizeGuided_inp_lq(self,inp_lq,x):
        B,C,H,W = x.size()
        return F.interpolate(inp_lq,size=(H,W))
    
    def forward(self, inp, time, guided_fts=None, inp_lq=None):

        timesteps = time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=inp.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(inp.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(inp.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t = self.time_mlp(timesteps)
        # print("Time embd: ", torch.any(torch.isnan(t)))
        # x is the guidance of x init concat with depth map
        x = inp
        # print(x.shape)
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        #print("Shape of x: ", x.shape)
        x = self.intro(x)
        # print("kernel weight: ", torch.any(torch.isnan(self.intro.weight.data)))
        # print("Input x: ",torch.any(torch.isnan(x)))
        encs = [x]
        i = 0

        for encoder, down in zip(self.encoders, self.downs):
            guided_fts_resized = self.ResizeGuided_fts(guided_fts,x)
            x, _, _ = encoder([x, t, guided_fts_resized])
                # print("Encoder Layer: " + str(i), torch.any(torch.isnan(x)))
            # print("Encoder Layer "+str(i)+" input shape: ", x.shape)
            encs.append(x)
            x = down(x)
            i+=1

        guided_fts_resized = self.ResizeGuided_fts(guided_fts,x)
        x, _, _ = self.middle_blks([x, t, guided_fts_resized])
        # print("Middle Layer: " , torch.any(torch.isnan(x)))
        i = 0
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            # print("Decoder Layer "+str(i)+" input shape: ", x.shape)
            x = x + enc_skip
            
            guided_fts_resized = self.ResizeGuided_fts(guided_fts,x)
            x, _, _ = decoder([x, t, guided_fts_resized])
            # print("Decoder Layer: " + str(i), torch.any(torch.isnan(x)))
            i+=1
           
            
        x = self.ending(x + encs[0])
        # x = x + inp[:, :3,...]
        # print("Ending x: ",torch.any(torch.isnan(x)) )
        x = x[..., :H, :W]
 
        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x 