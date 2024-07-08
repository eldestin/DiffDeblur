from distutils.command.clean import clean
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import torchvision.transforms as transforms
import os,sys
import random
from PIL import Image
from torchvision.utils import make_grid
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from RandomMask import *
# from add_gaussian import *
random.seed(2)
np.random.seed(2)

p = 1
AugDict = {
    1:tfs.ColorJitter(brightness=p),  #Brightness
    2:tfs.ColorJitter(contrast=p), #Contrast
    3:tfs.ColorJitter(saturation=p), #Saturation
    4:tfs.GaussianBlur(kernel_size=5), #Gaussian Blur
    #5:AddGaussianNoise(mean=0, std=30), #Gaussian Noise
    #5:RandomMaskwithRatio(64,patch_size=1,ratio=0.75), #Random Mask
}

class LFDOF(data.Dataset):
    '''
        This is the implementation of LFDOF dataset
    '''
    def __init__(self, path, train, size = 240, format='.png', crop = True):
        super(LFDOF, self).__init__()
        self.size = size
        print("crop size", size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'input'))
        # self.haze_img_dir_ls = [os.path.join(path, 'input', dir) for ]
        # print(self.haze_imgs_dir[:10])
        self.haze_imgs = [os.path.join(path, "input", a, img) for a in self.haze_imgs_dir for img in os.listdir(os.path.join(path, 'input', a))]
        print(len(self.haze_imgs))
        self.clear_dir=os.path.join(path,'ground_truth')
        self.crop = crop
    def __getitem__(self, index):
        haze_img = Image.open(self.haze_imgs[index])
        img_dir = self.haze_imgs[index]
        name_syn=img_dir.split('/')[-2]#.split('_')[0]#

        id = name_syn+".png"
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        if self.crop:
            i,j,h,w=tfs.RandomCrop.get_params(haze_img,output_size=(self.size,self.size))
            haze_img=FF.crop(haze_img,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
                      
        haze ,clear=self.augData(haze_img.convert("RGB") ,clear.convert("RGB"))
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
       
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        return  data , target
    def __len__(self):
        return len(self.haze_imgs)


class LFDOF_valid(data.Dataset):
    '''
        This is the implementation of LFDOF dataset
    '''
    def __init__(self, path, train, size = 240, format='.png', crop = True):
        super(LFDOF_valid, self).__init__()
        self.size = size
        print("crop size", size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'input'))
        # self.haze_img_dir_ls = [os.path.join(path, 'input', dir) for ]
        # print(self.haze_imgs_dir[:10])
        self.haze_imgs = [os.path.join(path, "input", a, img) for a in self.haze_imgs_dir for img in os.listdir(os.path.join(path, 'input', a))]
        print(len(self.haze_imgs))
        self.clear_dir=os.path.join(path,'ground_truth')
        self.crop = crop
        if self.crop:
            seed = 42
            random.seed(seed)
            img = Image.open(self.haze_imgs[0])
            i,j,h,w = tfs.RandomCrop.get_params(img,output_size=(self.size,self.size))
            self.info = [i,j,h,w]
    def __getitem__(self, index):
        haze_img = Image.open(self.haze_imgs[index])
        img_dir = self.haze_imgs[index]
        name_syn=img_dir.split('/')[-2]#.split('_')[0]#

        id = name_syn+".png"
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        if self.crop:
            i,j,h,w=self.info
            haze_img=FF.crop(haze_img,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
                      
        haze ,clear=self.augData(haze_img.convert("RGB") ,clear.convert("RGB"))
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
       
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        return  data , target
    def __len__(self):
        return len(self.haze_imgs)

class DPDD_cat(data.Dataset):
    '''
    This is the implementation of directed concat;
    '''
    def __init__(self, path, train, size = 240, format='.png', crop = True):
        super(DPDD_cat, self).__init__()
        self.size = size
        print("crop size", size)
        self.train = train
        self.format = format
        self.haze_imgs_dir_left=os.listdir(os.path.join(path,'inputL'))
        self.haze_imgs_dir_right=os.listdir(os.path.join(path,'inputR'))
        self.haze_imgs_l = [os.path.join(path,'inputL',img) for img in self.haze_imgs_dir_left]
        self.haze_imgs_r = [os.path.join(path,'inputR',img) for img in self.haze_imgs_dir_right]
        self.clear_dir=os.path.join(path,'target')
        self.crop = crop
    def __getitem__(self, index):
        haze_left = Image.open(self.haze_imgs_l[index])
        haze_right = Image.open(self.haze_imgs_r[index])

        img = self.haze_imgs_l[index]
        name_syn=img.split('/')[-1]#.split('_')[0]#
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        if self.crop:
            i,j,h,w=tfs.RandomCrop.get_params(haze_left,output_size=(self.size,self.size))
            haze_left=FF.crop(haze_left,i,j,h,w)
            haze_right=FF.crop(haze_right,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
          
            
        haze_r ,clear=self.augData(haze_right.convert("RGB") ,clear.convert("RGB"))
        haze_l, _ = self.augData(haze_left.convert("RGB") ,clear.convert("RGB"))
        haze_l = tfs.ToTensor()(haze_l)
        haze_r = tfs.ToTensor()(haze_r)
        clear = tfs.ToTensor()(clear)
       
        return haze_l,haze_r,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        return  data , target
    def __len__(self):
        return len(self.haze_imgs_l)

class DPDD_cat_valid(data.Dataset):
    '''
    This is the implementation of directed concat;
    '''
    def __init__(self, path, train, size = 240, format='.png', crop = True):
        super(DPDD_cat_valid, self).__init__()
        self.size = size
        print("crop size", size)
        self.train = train
        self.format = format
        self.haze_imgs_dir_left=os.listdir(os.path.join(path,'inputL'))
        self.haze_imgs_dir_right=os.listdir(os.path.join(path,'inputR'))
        self.haze_imgs_l = [os.path.join(path,'inputL',img) for img in self.haze_imgs_dir_left]
        self.haze_imgs_r = [os.path.join(path,'inputR',img) for img in self.haze_imgs_dir_right]
        self.clear_dir=os.path.join(path,'target')
        self.crop = crop
        if self.crop:
            seed = 42
            random.seed(seed)
            img = Image.open(self.haze_imgs_r[0])
            i,j,h,w = tfs.RandomCrop.get_params(img,output_size=(self.size,self.size))
            self.info = [i,j,h,w]
        self.crop = crop
    def __getitem__(self, index):
        
        haze_left = Image.open(self.haze_imgs_l[index])
        haze_right = Image.open(self.haze_imgs_r[index])

        img = self.haze_imgs_l[index]
        name_syn=img.split('/')[-1]#.split('_')[0]#
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        if self.crop:
            # i,j,h,w=tfs.RandomCrop.get_params(haze_left,output_size=(self.size,self.size))
            i,j,h,w = self.info
            haze_left=FF.crop(haze_left,i,j,h,w)
            haze_right=FF.crop(haze_right,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
          
            
        haze_r ,clear=self.augData(haze_right.convert("RGB") ,clear.convert("RGB"))
        haze_l, _ = self.augData(haze_left.convert("RGB") ,clear.convert("RGB"))
        haze_l = tfs.ToTensor()(haze_l)
        haze_r = tfs.ToTensor()(haze_r)
        clear = tfs.ToTensor()(clear)
       
        return haze_l,haze_r,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        return  data , target
    def __len__(self):
        return len(self.haze_imgs_l)

class DPDD_valid(data.Dataset):
    def __init__(self,path,train,size=240,format='.png',crop=True,mask_generator=None):
        super(DPDD_valid,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
      #  self.AugDict = AugDict
        self.haze_imgs_dir=os.listdir(os.path.join(path,'inputC'))
        #self.haze_imgs_dir = [x for x in haze_imgs_dir if ('.png' in x)]
        self.haze_imgs = [os.path.join(path,'inputC',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'target')   
        self.crop=crop
        if self.crop:
            seed = 2
            random.seed(seed)
            img = Image.open(self.haze_imgs[0])
            i,j,h,w = tfs.RandomCrop.get_params(img,output_size=(self.size,self.size))
            self.info = [i,j,h,w]
        
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        name_syn=img.split('/')[-1]#.split('_')[0]#
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
     
        if self.crop:
            i,j,h,w=self.info
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
          
            
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
       
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
                

        return  data , target
    def __len__(self):
        return len(self.haze_imgs)



class CUHK_blur(data.Dataset):
    '''
        The implementation of cuhk dataset
    '''
    def __init__(self,path,train,size=240,format='.jpg',crop=True,mask_generator=None):
        super(CUHK_blur,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
      #  self.AugDict = AugDict
        self.haze_imgs_dir=os.listdir(os.path.join(path,'image'))
        #self.haze_imgs_dir = [x for x in haze_imgs_dir if ('.png' in x)]
        self.haze_imgs = [os.path.join(path,'image',img) for img in self.haze_imgs_dir]
        self.crop=crop
        if self.crop:
            seed = 2
            random.seed(seed)
            img = Image.open(self.haze_imgs[0])
            i,j,h,w = tfs.RandomCrop.get_params(img,output_size=(self.size,self.size))
            self.info = [i,j,h,w]
        
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        name_syn=img.split('/')[-1]#.split('_')[0]#
        id = name_syn
        # clear_name=id
        # clear=Image.open(os.path.join(self.clear_dir,clear_name))
     
        if self.crop:
            i,j,h,w=self.info
            haze=FF.crop(haze,i,j,h,w)
            # clear=FF.crop(clear,i,j,h,w)
          
            
        # haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))

        haze = tfs.ToTensor()(haze)
        # clear = tfs.ToTensor()(clear)
       
        return haze,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
                

        return  data , target
    def __len__(self):
        return len(self.haze_imgs)

class Pixeldp(data.Dataset):
    '''
        The implementation of cuhk dataset
    '''
    def __init__(self,path,train,size=240,format='.jpg',crop=True,mask_generator=None):
        super(Pixeldp,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
      #  self.AugDict = AugDict
        self.haze_imgs_dir=os.listdir(os.path.join(path,'test_c'))
        #self.haze_imgs_dir = [x for x in haze_imgs_dir if ('.png' in x)]
        self.haze_imgs = [os.path.join(path,'test_c',img) for img in self.haze_imgs_dir]
        self.crop=crop
        if self.crop:
            seed = 2
            random.seed(seed)
            img = Image.open(self.haze_imgs[0])
            i,j,h,w = tfs.RandomCrop.get_params(img,output_size=(self.size,self.size))
            self.info = [i,j,h,w]
        
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        name_syn=img.split('/')[-1]#.split('_')[0]#
        id = name_syn
        # clear_name=id
        # clear=Image.open(os.path.join(self.clear_dir,clear_name))
     
        if self.crop:
            i,j,h,w=self.info
            haze=FF.crop(haze,i,j,h,w)
            # clear=FF.crop(clear,i,j,h,w)
          
            
        # haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))

        haze = tfs.ToTensor()(haze)
        # clear = tfs.ToTensor()(clear)
       
        return haze,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
                

        return  data , target
    def __len__(self):
        return len(self.haze_imgs)

class Realdof(data.Dataset):
    '''
        The implementation of cuhk dataset
    '''
    def __init__(self,path,train,size=240,format='.jpg',crop=True,mask_generator=None):
        super(Realdof,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
      #  self.AugDict = AugDict
        self.haze_imgs_dir=os.listdir(os.path.join(path,'source'))
        #self.haze_imgs_dir = [x for x in haze_imgs_dir if ('.png' in x)]
        self.haze_imgs = [os.path.join(path,'source',img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, "target")
        self.crop=crop
        if self.crop:
            seed = 2
            random.seed(seed)
            img = Image.open(self.haze_imgs[0])
            i,j,h,w = tfs.RandomCrop.get_params(img,output_size=(self.size,self.size))
            self.info = [i,j,h,w]
        
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        name_syn=img.split('/')[-1]#
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
     
        if self.crop:
            i,j,h,w=self.info
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
          
            
        # haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))

        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
       
        return haze,clear,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
                

        return  data , target
    def __len__(self):
        return len(self.haze_imgs)


class DPDD(data.Dataset):
    def __init__(self,path,train,size=240,format='.png',crop=True,mask_generator=None, name = False):
        super(DPDD,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
      #  self.AugDict = AugDict
        self.haze_imgs_dir=os.listdir(os.path.join(path,'inputC'))
        #self.haze_imgs_dir = [x for x in haze_imgs_dir if ('.png' in x)]
        self.haze_imgs = [os.path.join(path,'inputC',img) for img in self.haze_imgs_dir]
        self.depth_path = "/hpc2hdd/home/hfeng108/Archive/models/Alignment/save_img"
        self.clear_dir=os.path.join(path,'target')
        self.name = name
        self.crop=crop
        
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        name_syn=img.split('/')[-1]#.split('_')[0]#
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        depth = Image.open(os.path.join(self.depth_path, clear_name))
        if self.crop:
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
            depth=FF.crop(depth,i,j,h,w)
          
            
        haze,clear, depth=self.augData(haze.convert("RGB") ,clear.convert("RGB"), depth)
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
        depth = tfs.ToTensor()(depth)
        if self.name:
            return haze,clear, clear_name
        else:
            return haze, clear, depth
    def augData(self,data,target, depth):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            depth = tfs.RandomHorizontalFlip(rand_hor)(depth)
            # if rand_rot:
            #     data=FF.rotate(data,90*rand_rot)
            #     target=FF.rotate(target,90*rand_rot)
            #     depth = FF.rotate(depth, 90*rand_rot)
                

        return  data , target, depth
    def __len__(self):
        return len(self.haze_imgs)
