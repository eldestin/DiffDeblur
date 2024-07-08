import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig
)
from transformers import SegformerDecodeHead
import warnings
import torch.nn.functional as F
import torch.nn as nn 
from transformers import AutoModel, AutoConfig
from transformers import SegformerFeatureExtractor, SegformerForImageClassification

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
     

class SegFormer(nn.Module):
    def __init__(self, pretrianed=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        seg_model_path=r'/hpc2hdd/home/hfeng108/Archive/segformer/model'
        self.encoder = Segformer_Class= SegformerForImageClassification.from_pretrained(seg_model_path)
        self.head_config = AutoConfig.from_pretrained("/hpc2hdd/home/hfeng108/Archive/segformer/decoder_config/config.json")
        self.head_config.num_labels = 1
        self.decoder = SegformerDecodeHead._from_config(self.head_config)
    
    def forward(self, input):
        _,_,H,W = input.size()
        output_feats = self.encoder(input,output_hidden_states=True)
        outputs = self.decoder(output_feats.hidden_states)
        outputs_resized = resize(
            input=outputs,
            size=(H,W),
            mode='bilinear',
            align_corners=False)
        return outputs_resized

if __name__ == '__main__':
    from transformers import SegformerFeatureExtractor, SegformerForImageClassification


    model = SegformerForImageClassification.from_pretrained("/home/haipeng/Code/shadow/hgf_pretrain/nvidia/mit-b0")

    inputs = torch.randn(4,3,256,256)
    outputs = model(inputs,output_hidden_states=True)
    print("adasd")