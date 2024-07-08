import torch
from tqdm import tqdm
from functools import partial

from diffusers import DDIMScheduler
from contextlib import contextmanager

class SR3scheduler(DDIMScheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001,beta_end: float = 0.02, beta_schedule: str = 'linear',diff_chns=3, prediction_type = "epsilon"):
        super().__init__(num_train_timesteps, beta_start ,beta_end,beta_schedule, prediction_type = prediction_type)
        # Initialize other attributes specific to SR3scheduler class
        # ...
        self.diff_chns = diff_chns
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Only modify the last three channels of the tensor (assuming channels are in the second dimension)
        num_channels = original_samples.shape[1]
        if num_channels > self.diff_chns:
            original_samples_select = original_samples[:, -self.diff_chns:].contiguous()
            noise_select = noise[:, -self.diff_chns:].contiguous()

            noisy_samples_select = sqrt_alpha_prod * original_samples_select + sqrt_one_minus_alpha_prod * noise_select
           # print("Add_noise shape: ", noisy_samples_select.shape)
            noisy_samples = original_samples.clone()
            noisy_samples[:, -self.diff_chns:] = noisy_samples_select
        else:
            noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

def create_SR3scheduler(opt,phase):
    
    steps= opt['num_train_timesteps'] if phase=="train" else opt['num_test_timesteps']
    scheduler=SR3scheduler(
        num_train_timesteps = steps,
        beta_start = opt['beta_start'],
        beta_end = opt['beta_end'],
        beta_schedule = opt['beta_schedule'],
        prediction_type = opt['prediction_type']
    )
    return scheduler
    
class SR3Sampler():
    
    def __init__(self,model: torch.nn.Module,scheduler:SR3scheduler,eta: float = 0.0):
        self.model = model
        self.scheduler = scheduler
        self.eta = eta
    
    def sample_high_res(self,low_res: torch.Tensor, conditions=None,label=None,guided_fts=None,inp_lq=None,show_pbar=True):
        "Using Diffusers built-in samplers"
        device = next(self.model.parameters()).device
       # print(self.eta)
        eta = torch.Tensor([self.eta]).to(device)
        #print("x_cond shape: ", low_res.shape)
        if low_res.shape[1] == 3:
            HR_image = torch.randn_like(low_res, device=device)
        elif low_res.shape[1] == 1:
            # if condition channel is 1, disparity map
            HR_image = torch.randn(low_res.shape[0], 3, low_res.shape[2], low_res.shape[3], device=device)
        elif low_res.shape[1] == 4:
            # condition channel 12
            # print("shape 5")
            HR_image = torch.randn(low_res.shape[0], 3, low_res.shape[2], low_res.shape[3], device=device)
        else:
            HR_image = torch.randn_like(low_res, device=device)
            HR_image = HR_image[:, low_res.shape[1]//2:,:,:]
        # print(HR_image.shape)
        low_res=low_res.to(device)
        preds = []
        if show_pbar:
            pbar = tqdm(total=len(self.scheduler.timesteps))
        for t in self.scheduler.timesteps:
            if show_pbar:
                pbar.set_description(f"DDIM Sampler: frame {t}")
            self.model.eval()
            with torch.no_grad():
                if conditions is not None:
                    batch_data = conditions+[low_res,HR_image]
                else:
                    batch_data=[low_res,HR_image]
                if (guided_fts is not None) and (label is not None):
                    noise = self.model(torch.cat(batch_data, dim=1), t, label, guided_fts)
                elif guided_fts is not None and (inp_lq is not None):
                    noise = self.model(torch.cat(batch_data, dim=1), t, guided_fts, inp_lq)
                elif guided_fts is not None:
                    noise = self.model(torch.cat(batch_data, dim=1), t, guided_fts)
                elif label is not None:
                    noise = self.model(torch.cat(batch_data, dim=1), t, label)
                else:
                    noise = self.model(torch.cat(batch_data, dim=1), t)
            assert noise.size()==HR_image.size()
           # print("noise:",noise.size())
            HR_image = self.scheduler.step(model_output = noise,timestep = int(t),  sample = HR_image, eta=eta).prev_sample #eta = eta
            # if t % int(len(self.scheduler.timesteps) / 7) == 0 or t == len(self.scheduler.timesteps):
            #     preds.append(HR_image.detach().float().cpu())
            if show_pbar:
                pbar.update(1)
            del noise
            torch.cuda.empty_cache()
        return HR_image,preds
    
def create_SR3Sampler(model,opt):
    
    scheduler = create_SR3scheduler(opt,"test")
    scheduler.set_timesteps(opt['num_test_timesteps'])
    sampler = SR3Sampler(
        model = model,
        scheduler = scheduler,
        eta = opt['eta']
    )
    return sampler
