'''
This implementation is the summarize of all improve strategy works in single image deblurring.
The component can be summarized as follows:
    1. Res50 instead Res18;
    2. semantic code as guidance;
    3. initial predictor： predict the sample image first, then utilize residual to diffusion;
    4. the diffusion process predict image instead of nosie;
'''
import sys
sys.path.append("../..")
from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from diffusers.models.unet_2d import UNet2DModel as Unet
import os
import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils.imgqual_utils import SSIM
from utils.imgqual_utils import PSNR_DPDD as PSNR
from utils.optimizer import get_cosine_schedule_with_warmup
from torchvision.utils import save_image
from kornia.color import rgb_to_grayscale
from easydict import EasyDict
#from ema import LitEma
from loss.CL1 import PSNRLoss
# from models.DenoisingNetwork.DenoisingNAFNet_arch import NAFNet as DenoisingNAFNet
from models.DenoisingNetwork.DenoisingNetwork import NAFNetFusionModifiedcat as DenoisingNAFNet
from models.DenoisingNetwork.DenoisingNetwork import resnet50
from models.Segmodels.segformer_mit import SegFormer
import pipeline
from contextlib import contextmanager
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('medium')
import lpips
from utils.BlurDataset import DPDD, LFDOF, DPDD_cat, DPDD_cat_valid,DPDD_valid, CUHK_blur, Realdof
from utils.EMA import EMA as EMA_callback
from collections import OrderedDict
from models.DenoisingNetwork.Predelur_network import NAFNetModified as DenoisingNAFNetInitial
from torchmetrics.image.fid import FrechetInceptionDistance
# NCCL_SOCKET_IFNAME=eth2
class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.max_steps = self.params.Trainer.max_steps
        self.save_path='/hpc2hdd/home/hfeng108/Archive/save_image/save_single'
        
        
        # Add segformer for disparity map estimation guidance
        self.depth_model = SegFormer()
        # load path from pretrain 
        # change to real dataset to test again
        # depth_path = "/home/fhx/code/archive/models/DEM_Segformer/logs/Seg_Logs/synthetic/checkpoints/Seg-epoch=94-psnr=22.814829-ssim=0.843821.ckpt"
        # Real image pretrain segformer
        depth_path = "/hpc2hdd/home/hfeng108/Archive/models/DEM_Segformer/logs/Seg_Logs/version_0/checkpoints/Seg-epoch=94-iou=0.884012.ckpt"
        # checkpoint = torch.load(self.depth_path)
        #print(checkpoint["state_dict"])
        ckpt = self.convert_pl1(depth_path)
        print("Load pretrained depth model: ")
        # print("Load segmentation data: ")
        self.depth_model.load_state_dict(ckpt)
        # self.fid = FrechetInceptionDistance(feature=64)
        
        self.initlr = self.params.Trainer.initlr
        self.train_datasets = self.params.Trainer.train_datasets
        self.train_batchsize = self.params.Trainer.train_bs
        self.validation_datasets = self.params.Val.val_datasets
        self.val_batchsize = self.params.Val.val_bs
        self.val_crop = True

    
        #Train setting
        self.initlr = self.params.Trainer.initlr #initial learning
        self.crop_size = self.params.Trainer.crop_size #random crop size
        self.num_workers = self.params.Trainer.num_workers
      
                
        print('validation_datasets:',self.validation_datasets)
        print('training num:',self.train_dataloader().__len__())
        print('validation num:',self.val_dataloader().__len__())
        
        # align_settings = self.params.Alignformer.settings
        # self.align_module = Alignformer(**align_settings)
        self.diff_opt = self.params.diff_opt
        # self.loss_F = L1_Charbonnier_loss()
        self.loss_F = PSNRLoss() #self.noise_estimation_los
        # self.loss_F = VGG_loss()
        #self.loss_per = PerceptualLoss2()
        model_settings = self.params.Model.settings
        # model_settings.Finetune = True
        model_finetune = DenoisingNAFNet(
            **model_settings
        )
        self.semantic_encoder = resnet50()
        self.model = model_finetune
        
    
        self.lpips_fn = lpips.LPIPS(net='alex')
        self.threshold = 0.5
        self.model_initial = DenoisingNAFNetInitial(**self.params.InitModel.settings)
        # self.depth_predictor = DenoisingNAFNetInitial(**self.params.DepthPredictor.settings)
        self.freeze_dpt = self.params.freeze_dpt
        self.update_lr = self.params.updatelr
        self.flag = 0
        self.use_ema = False
        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train')
        )
        self.DiffSampler.scheduler.set_timesteps(self.diff_opt['scheduler']['num_test_timesteps'])  
        
        
        self.automatic_optimization = False
        self.pred_samples = []
        self.gt_samples = []
        self.save_hyperparameters()
        self.mae = nn.L1Loss()
    def setup_seeds(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def convert_pl(self, path):
        '''
        This function aims to convert PT lightning parameters dictionary into torch load state dict 
        '''
        ckpt = torch.load(path,map_location='cpu')
        # print(ckpt['state_dict'].keys())
        new_state_dict = OrderedDict()
        for k in ckpt['state_dict']:
            # print(k)
            #name = 1
            # print(k[:6])
            if k[:6] != 'model.':
            #if 'tiny_unet.' not in k:
                continue
            if k[:6] == 'model.':
                name = k.replace('model.','')
            # print(name)
            new_state_dict[name] = ckpt['state_dict'][k]
        return new_state_dict

    def convert_pl1(self, path):
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
    
    def freezy_params(self,module):
        for name,params in module.named_parameters():
            params.requires_grad = False
    def unfreezy_params(self,module):
        for name,params in module.named_parameters():
            params.requires_grad = True
    
    def lpips_score_fn(self,x,gt):
        self.lpips_fn.to(self.device)
        lp_score = self.lpips_fn(
            gt * 2 - 1, x * 2 - 1
        )
        return torch.mean(lp_score).item()
    
    def add_fid(self, pred, gt):
        self.fid.update(pred, real = False)
        self.fid.update(gt, real = True)

    def configure_optimizers(self):
        # REQUIRED
        parameters=[
            # {'params': self.depth_predictor.parameters()},
            {'params': self.model_initial.parameters()},
            {'params':self.model.parameters()},
           {'params':self.semantic_encoder.parameters()}
        ]
        print(filter(lambda p: p.requires_grad, self.model.parameters()))
        #optimizer = Lion(parameters, lr=self.initlr,betas=[0.9,0.99])    
        #optimizer = torch.optim.Adam(parameters, lr=self.initlr, weight_decay=0.000,eps=0.00000001,betas=(0.9, 0.999), amsgrad=False)    
        optimizer = torch.optim.AdamW(parameters, lr=self.initlr,betas=[0.9,0.999], eps = 1e-6, weight_decay=0.01)
        scheduler2 = get_cosine_schedule_with_warmup(optimizer, 200,self.max_steps) #self.max_steps*0.02,
        
        return [optimizer], [scheduler2]

    def training_epoch_start(self):
        self.scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train')
        # self.loss_F = VGG_loss()
    
    def on_train_start(self):
        print("Seeds setup: ")
        # self.setup_seeds(42)
        pass
        #self.loss_F = VGG_loss().to(torch.float16).cuda()
    
    def on_validation_start(self):
        print("Seeds setup: ")
        self.setup_seeds(15)
        self.fid = FrechetInceptionDistance(feature=192).to(self.device)
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
    
    def max_min_normalize(self, inp):
        inp_norm = (inp - inp.min()) / (inp.max() - inp.min())
        return inp_norm
    
    def on_train_epoch_start(self):
        self.flag += 1
    def update_lrate(self):
        opt = self.trainer.strategy._lightning_optimizers
        for param in opt[0].state_dict()["opt"]["opt"]["param_groups"]:
            
            param["lr"] = 0.002
            print("Param dict: ", param)
            # print(self.trainer.strategy._lightning_optimizers[0].state_dict())
            # print("Update lr",param["lr"])
        self.update_lr = False
        self.trainer.strategy._lightning_optimizers = opt
        print(self.trainer.strategy._lightning_optimizers[0])
        #return opt
    def training_step(self, batch, batch_idx):
        # print(self.trainer.strategy._lightning_optimizers)
        # if self.update_lr:
        #     self.update_lrate()
        opt = self.optimizers()
        #print("Keys: ",opt.state_dict()["opt"]["opt"]["param_groups"].keys() )  
        # print("Current lr: ", opt.state_dict()["opt"]["opt"]["param_groups"][0]["lr"])
        opt.zero_grad()
        
        x, gt, gt_guide = batch
        B,C,H,W = gt.size()     
        # fix depth model
        # blur image as input, output the blur map estimation
        with torch.no_grad():
            # blur map
            dpt = self.max_min_normalize(self.depth_model(x))
        # directly concat as guidance input
        inp = torch.cat([x, dpt], dim = 1)


        # Got predeblur image
        x_init = self.model_initial(inp)
        
        # Got residual 
        residual_gt = gt - x_init

        # with torch.no_grad():
        # got semantic guidance -> res50
        semantic_code = self.semantic_encoder(x_init)
        with torch.no_grad():
            # get predeblurred image defocus map
            init_blur_guide = self.max_min_normalize(self.depth_model(x_init))
            # gt_guide = self.max_min_normalize(self.depth_model(gt))
            # gt_guide_res = gt_guide - init_blur_guide
        # Train Depth_predictor to predict gt with blur input
        # init_guide_res = self.depth_predictor(init_blur_guide)
        # Residual learning
        init_guide = init_blur_guide

        # condition on predeblurred image, map;
        batch = torch.cat([x_init, init_guide ,residual_gt], dim = 1)
    
        noise = torch.randn(batch.shape).to(self.device)
        bs = batch.shape[0]
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bs,), device=self.device).long()
        noisy_images = self.DiffSampler.scheduler.add_noise(batch, timesteps=timesteps, noise=noise)
        #print("timesteps:",timesteps)#.type())
        
        sample_pred = self.model(noisy_images, timesteps, semantic_code)
        # with torch.autograd.detect_anomaly():
        # loss_depth = self.loss_F(gt_guide_res,init_guide_res)
        loss = self.loss_F(residual_gt, sample_pred)
        # loss = self.loss_F(gt - x, sample_pred + (x_init-x))
        # loss = self.loss_F(gt - x_init) + self.loss_F(residual_gt, sample_pred)
        self.manual_backward(loss)
        #clip grad norm
        #before clipping:
        # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        # with torch.no_grad():
        #     total_norm = 0.0
        #     for p in self.model.parameters():
        #         param_norm = p.grad.detach().data.norm(2)
        #         total_norm += param_norm.item() ** 2
        #     total_norm = total_norm ** 0.5
            # print("Model total norm After clipping: ", total_norm)

        # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10000)
        # print("After clipping Norm: ",total_norm)
        opt.step()
        sch = self.lr_schedulers()
        sch.step()
        
        with torch.no_grad():
            # samples = sample_pred + x_init
            # psnr = PSNR(samples.float(),gt.float())
            # ssim = SSIM(samples.float(),gt.float())
            # lpips_score = self.lpips_score_fn(samples.float(),gt.float())
            self.log("train_loss",loss,prog_bar=True)
            # self.log("log_depth", loss_depth, prog_bar=True) 
            # self.log("train_psnr", psnr, prog_bar=True)
            # self.log("train_ssim", ssim, prog_bar=True)
            # self.log("train_lpips_score", lpips_score, prog_bar=True)
            # self.log("Total norm", total_norm, prog_bar=True)
        return {"loss":loss}
    
    def on_validation_epoch_start(self) -> None:
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            print("Switched to EMA weights")

    def on_validation_epoch_end(self) -> None:

        if self.use_ema:
            self.model_ema.restore(self.model.parameters())
            print("Restored training weights")

    def validation_step(self,batch,batch_idx):
        
        x, gt = batch
        #print(gt)
        B,C,H,W = gt.size() 
        print(B,C,H,W)       
        # Extract disparity map
        dpt = self.max_min_normalize(self.depth_model(x))
        # directly concat as guidance input
        inp = torch.cat([x, dpt], dim = 1)
        x_init = self.model_initial(inp)
        
        # Got residual 
        residual_gt = gt - x_init
        init_blur_guide = self.max_min_normalize(self.depth_model(x_init))
        # init_guide_res = self.depth_predictor(init_blur_guide)
        init_guide = init_blur_guide
        # utilize align result as semantic code
        # also generate new depth map guidance for initial output
        # with torch.no_grad():
        semantic_code = self.semantic_encoder(x_init)
        samples_residual,preds = self.DiffSampler.sample_high_res(torch.cat([x_init, init_guide], dim = 1), guided_fts = semantic_code)
        samples = x_init + samples_residual
        

        psnr_init = PSNR(x_init.float(),gt.float())
        ssim_init = SSIM(x_init.float(),gt.float())
        lpips_score_init = torch.tensor(self.lpips_score_fn(x_init.float(),gt.float()))
        # fid_init = self.cal_fid(x_init.to(torch.uint8), gt.to(torch.uint8))

        psnr = PSNR(samples.float(),gt.float())
        ssim = SSIM(samples.float(),gt.float())
        lpips_score =  torch.tensor(self.lpips_score_fn(samples.float(),gt.float()))
        # print(samples)
        self.pred_samples.append((samples*255).to(torch.uint8))
        self.gt_samples.append((gt*255).to(torch.uint8))
        fid = self.add_fid((samples*255).to(torch.uint8), (gt*255).to(torch.uint8))
        mae = self.mae(samples, gt)
        # if batch_idx==0:
        filename = "pred_{}.png".format(batch_idx)
        save_image(samples[...],os.path.join(self.save_path, filename))      
    # if batch_idx==0:
        filename = "x_init_{}.png".format(batch_idx)
        save_image(x_init[:2,...],os.path.join(self.save_path, filename))
    # if batch_idx==0:
        filename = "init_guide{}.png".format(batch_idx)
        save_image(init_guide[:2,...],os.path.join(self.save_path, filename))
    # if batch_idx==0:
    #     filename = "x_align_{}.png".format(self.current_epoch)
    #     save_image(x[:4,...],os.path.join(self.save_path, filename))
    # if batch_idx==0:
        filename = "target_{}.png".format(batch_idx)
        save_image(gt[:2,...],os.path.join(self.save_path, filename))
        # if batch_idx==0:
        #     filename = "syn_input_{}.png".format(self.current_epoch)
        #     save_image(syn_input[:4,...],os.path.join(self.save_path, filename))

        # add test loss
        with torch.no_grad(): 
            test_loss = self.loss_F(residual_gt, samples_residual)
        test_loss = test_loss.to(self.device)
        psnr_init = psnr_init.to(self.device)
        ssim_init=ssim_init.to(self.device)
        lpips_score_init=lpips_score_init.to(self.device)
        psnr = psnr.to(self.device)
        ssim = ssim.to(self.device)
        lpips_score = lpips_score.to(self.device)
        
        self.log("test_loss", test_loss, sync_dist=True)
        self.log('psnr_init',psnr_init, sync_dist=True)
        self.log('ssim_init',ssim_init, sync_dist=True)
        # self.log("fid_init", fid_init, sync_dist = True)
        self.log('lpips_score_init',lpips_score_init, sync_dist=True)
        self.log('psnr',psnr, sync_dist=True)
        self.log('ssim',ssim, sync_dist=True)
        self.log('lpips_score',lpips_score, sync_dist=True)
        self.log("mae", mae, sync_dist=True)
        # self.log("fid_score", fid, sync_dist = True)    

        return {"psnr":psnr,"ssim":ssim}
    
    
    def train_dataloader(self):
        
        train_set = DPDD(self.train_datasets,train=True,size=self.crop_size)
        train_loader = DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers)

        return train_loader
    
    def val_dataloader(self):
        val_set = DPDD_valid(self.validation_datasets,train=False,size=200,crop=False)
        val_loader = DataLoader(val_set, batch_size=self.val_batchsize, shuffle=False, num_workers=self.num_workers)

        return val_loader

    def init_params(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            print("Start initial layer: ", m)


def main(): 
    print("Test without pos enc")
    Val=False
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters = True)
    config_path = r'/option/NAFNetAlign_guidance.yaml'
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)
    model = CoolSystem(config)
    if config.RESUME == False:
        resume_checkpoint_path = None
    else:
        resume_checkpoint_path = config.resume_checkpoint_path
    
    print("Load Pretrain model: ")
    # path = "/hpc2hdd/home/hfeng108/Archive/lgs/NAFNetguidancecat/Defocus_delburring_initial/version_30/checkpoints/epoch04-PSNR-26.689-SSIM-0.8195-lpips_score-0.1632.ckpt"
    # model = CoolSystem.load_from_checkpoint(path, hparams = config)
    checkpoint_callback = ModelCheckpoint(
    monitor='psnr',
    filename='epoch{epoch:02d}-PSNR-{psnr:.3f}-SSIM-{ssim:.4f}-lpips_score-{lpips_score:.4f}',
    auto_insert_metric_name=False,   
    every_n_epochs=1,
    save_top_k=8,
    mode = "max",
    save_last=True
    )
    ema_ck = EMA_callback(decay = 0.999)
    output_dir = '/hpc2hdd/home/hfeng108/Archive/lgs/NAFNetguidancecat'
    version_name='Baseline'
    logger = TensorBoardLogger(name=config.log_name,save_dir = output_dir )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        check_val_every_n_epoch=config.Trainer.check_val_every_n_epoch,
        max_steps=config.Trainer.max_steps,
        accelerator=config.Trainer.accelerator,
        devices=config.Trainer.devices,
        precision=config.Trainer.precision,
        accumulate_grad_batches = config.Trainer.accumulate_grad_batches,
        logger=logger,
        strategy=ddp,
        enable_progress_bar=True,
        log_every_n_steps=config.Trainer.log_every_n_steps,
        callbacks = [checkpoint_callback,lr_monitor_callback, ema_ck]
    )
    trainer.fit(model,ckpt_path=None)
        
    
if __name__ == '__main__':
	#your code 
    main()
