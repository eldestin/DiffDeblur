import yaml
from easydict import EasyDict
from collections import OrderedDict 
import torch
def load_sub_model_ckpt(model,sub_model,sub_model_ckpt_path):
    ckpt = torch.load(sub_model_ckpt_path,map_location='cpu')
    sub_model.load_state_dict(ckpt)
    sub_model_state_dict = sub_model.state_dict()
    model_state_dict = model.state_dict()
    new_model_state_dict = OrderedDict()
    for key in model_state_dict.keys():
        if key in sub_model_state_dict:
            new_model_state_dict[key] = sub_model_state_dict[key]
        else:
            new_model_state_dict[key] = model_state_dict[key]
    model.load_state_dict(new_model_state_dict)
    return model