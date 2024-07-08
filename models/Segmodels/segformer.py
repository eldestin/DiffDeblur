import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig
)


# reconstruct segformer in shadow detection
def segformer(pretrained=True):
    id2label = {0: "others"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "/home/yetian/Project/pl-sem-seg/models/segformer/nvidia-b4-finetuned-ade-512-512",
            ignore_mismatched_sizes=True,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id)
        return model

    else :
        config = SegformerConfig.from_json_file("/home/yetian/Project/pl-sem-seg/models/segformer/nvidia-b3-finetuned-ade-512-512/config.json")
        config.num_labels = num_labels
        config.id2label = id2label
        config.label2id = label2id
        model = SegformerForSemanticSegmentation(config)
        return model 

import torch.nn as nn 

class SegFormer(nn.Module):
    def __init__(self, pretrianed=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = segformer(pretrianed)

    
    def forward(self, x):
        logits = self.model(x).logits
        upsampled_logits = nn.functional.interpolate(
                logits, scale_factor=4.0, mode="bilinear", align_corners=False
            )
        return upsampled_logits


if __name__ == '__main__':
    from transformers import SegformerFeatureExtractor, SegformerForImageClassification


    model = SegformerForImageClassification.from_pretrained("/home/haipeng/Code/shadow/hgf_pretrain/nvidia/mit-b0")

    inputs = torch.randn(4,3,256,256)
    outputs = model(inputs,output_hidden_states=True)
    print("adasd")