import os
from collections import OrderedDict
import numpy as np
import mcubes
import omegaconf
from termcolor import colored
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F

import torch
from torch import nn, optim

import torchvision.utils as vutils

from models.base_model import BaseModel
from models.networks.triplane_vae_network import TriplaneVAE

import utils.util
from utils.util_3d import init_mesh_renderer, render_sdf
from utils.distributed import reduce_loss_dict


class TriplaneVAEModel(BaseModel):
    def name(self):
        return 'TriplaneVAE-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device

        assert opt.vq_cfg is not None
        configs = omegaconf.OmegaConf.load(opt.vq_cfg)
        mparam = configs.model.params
        
        resolution = mparam.ddconfig.resolution
        z_channels = mparam.ddconfig.z_channels
        base_channels = mparam.ddconfig.ch
        n_downsamples = len(mparam.ddconfig.ch_mult) - 1
        
        self.vae = TriplaneVAE(
            in_channels=1,
            z_channels=z_channels,
            resolution=resolution,
            base_channels=base_channels,
            hidden_dim=256,
            n_downsamples=n_downsamples
        )
        self.vae.to(self.device)
        
        if self.isTrain:
            self.kl_weight = configs.lossconfig.params.get('kl_weight', 1e-6)
            self.recon_weight = configs.lossconfig.params.get('recon_weight', 1.0)
            
            self.optimizer = optim.Adam(
                self.vae.parameters(), 
                lr=opt.lr, 
                betas=(0.5, 0.9)
            )
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)
            
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]
            
            self.print_networks(verbose=False)
        
        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)
        
        if 'snet' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif opt.dataset_mode == 'buildingnet':
            dist, elev, azim = 1.0, 20, 20
        else:
            dist, elev, azim = 1.7, 20, 20
            
        self.renderer = init_mesh_renderer(
            image_size=256, 
            dist=dist, 
            elev=elev, 
            azim=azim, 
            device=self.device
        )
        
        self.best_iou = -1e12
        
        if self.opt.distributed:
            self.make_distributed(opt)
            self.vae_module = self.vae.module
        else:
            self.vae_module = self.vae
    
    def switch_eval(self):
        self.vae.eval()
    
    def switch_train(self):
        self.vae.train()
    
    def make_distributed(self, opt):
        self.vae = nn.parallel.DistributedDataParallel(
            self.vae,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
    
    def set_input(self, input):
        self.x = input['sdf']
        self.cur_bs = self.x.shape[0]
        vars_list = ['x']
        self.tocuda(var_names=vars_list)
    
    def forward(self):
        self.x_recon, self.mu, self.logvar = self.vae(self.x)
    
    @torch.no_grad()
    def inference(self, data, should_render=False, verbose=False):
        self.switch_eval()
        self.set_input(data)
        
        with torch.no_grad():
            self.z = self.vae.encode_no_sample(self.x)
            self.x_recon = self.vae.decode_no_sample(self.z)
            
            if should_render:
                self.image = render_sdf(self.renderer, self.x)
                self.image_recon = render_sdf(self.renderer, self.x_recon)
        
        self.switch_train()
    
    def test_iou(self, data, thres=0.0):
        self.inference(data, should_render=False)
        
        x = self.x
        x_recon = self.x_recon
        
        iou = utils.util.iou(x, x_recon, thres)
        return iou
    
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        
        iou_list = []
        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):
                iou = self.test_iou(test_data, thres=thres)
                iou_list.append(iou.detach())
        
        iou = torch.cat(iou_list)
        iou_mean, iou_std = iou.mean(), iou.std()
        
        ret = OrderedDict([
            ('iou', iou_mean.data),
            ('iou_std', iou_std.data),
        ])
        
        if ret['iou'] > self.best_iou:
            self.best_iou = ret['iou']
            save_name = f'epoch-best'
            self.save(save_name, global_step)
        
        self.switch_train()
        return ret
    
    def compute_kl_loss(self, mu, logvar):
        # KL(N(mu, sigma^2) || N(0, 1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return kl_loss.mean()
    
    def backward(self):
        recon_loss = F.mse_loss(self.x_recon, self.x, reduction='mean')
        
        kl_loss = self.compute_kl_loss(self.mu, self.logvar)
        
        self.loss_rec = recon_loss
        self.loss_kl = kl_loss
        self.loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        
        loss_dict = {
            'loss_total': self.loss.clone().detach(),
            'loss_rec': self.loss_rec.clone().detach(),
            'loss_kl': self.loss_kl.clone().detach(),
        }
        self.loss_dict = reduce_loss_dict(loss_dict)
        
        self.loss.backward()
    
    def optimize_parameters(self, total_steps):
        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()
    
    def get_current_errors(self):
        ret = OrderedDict([
            ('total', self.loss_dict['loss_total'].mean().data),
            ('rec', self.loss_dict['loss_rec'].mean().data),
            ('kl', self.loss_dict['loss_kl'].mean().data),
        ])
        return ret
    
    def get_current_visuals(self):
        try:
            with torch.no_grad():
                self.image = render_sdf(self.renderer, self.x)
                self.image_recon = render_sdf(self.renderer, self.x_recon)
            vis_tensor_names = ['image', 'image_recon']
            vis_ims = self.tnsrs2ims(vis_tensor_names)
            return OrderedDict(zip(vis_tensor_names, vis_ims))
        except Exception as e:
            print(f"[warn] visualization skipped: {e}")
            return OrderedDict()
 
    def save(self, label, global_step=0, save_opt=False):
        state_dict = {
            'vae': self.vae_module.state_dict(),
            'global_step': global_step,
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()
        
        save_filename = 'triplane_vae_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)
        
        torch.save(state_dict, save_path)
        print(colored(f'[*] Model saved to: {save_path}', 'blue'))
    
    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
        
        if 'vae' in state_dict:
            self.vae.load_state_dict(state_dict['vae'])
        else:
            self.vae.load_state_dict(state_dict)
        
        print(colored(f'[*] Weight successfully loaded from: {ckpt}', 'blue'))
        
        if load_opt and 'opt' in state_dict:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored(f'[*] Optimizer restored from: {ckpt}', 'blue'))
