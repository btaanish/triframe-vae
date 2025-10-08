import os
from collections import OrderedDict
import cv2
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils

from models.base_model import BaseModel
from models.networks.triplane_vae_network import TriplaneVAE
from models.networks.diffusion_networks.network import DiffusionUNet
from models.networks.bert_networks.network import BERTTextEncoder
from models.networks.edm_diffusion_utils import (
    EDMPrecond, EDMLoss, EDMSampler, 
    get_edm_schedule_params
)

from utils.distributed import reduce_loss_dict
from utils.util_3d import init_mesh_renderer, render_sdf


class EDMTriplaneText2ShapeModel(BaseModel):
    def name(self):
        return 'EDM-Triplane-Text2Shape-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device

        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        mparam = vq_conf.model.params
        ddconfig = mparam.ddconfig
        resolution = ddconfig.resolution
        z_channels = ddconfig.z_channels
        base_channels = ddconfig.ch
        n_down = len(ddconfig.ch_mult) - 1
        
        self.triplane_vae = TriplaneVAE(
            in_channels=1,
            z_channels=z_channels,
            resolution=resolution,
            base_channels=base_channels,
            hidden_dim=256,
            n_downsamples=n_down
        )
        self.triplane_vae.to(self.device)
        
        if opt.vq_ckpt is not None:
            self.load_vae_ckpt(opt.vq_ckpt)
            print(colored(f'[*] Loaded VAE from {opt.vq_ckpt}', 'green'))
        
        for param in self.triplane_vae.parameters():
            param.requires_grad = False
        self.triplane_vae.eval()
        
        triplane_dim = resolution // (2 ** n_down)
        self.z_shape = (3 * z_channels, triplane_dim, triplane_dim)
        
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        
        unet_params['dims'] = 2  
        unet_params['in_channels'] = 3 * z_channels
        unet_params['out_channels'] = 3 * z_channels
        
        self.base_diffusion = DiffusionUNet(
            unet_params, 
            vq_conf=vq_conf, 
            conditioning_key=df_model_params.conditioning_key
        )
        self.base_diffusion.to(self.device)
        
        edm_params = get_edm_schedule_params()
        self.diffusion = EDMPrecond(
            self.base_diffusion,
            sigma_data=edm_params['sigma_data'],
            sigma_min=edm_params['sigma_min'],
            sigma_max=edm_params['sigma_max']
        )
        self.diffusion.to(self.device)
        
        self.edm_params = edm_params
        self.sigma_data = edm_params['sigma_data']
        self.sigma_min = edm_params['sigma_min']
        self.sigma_max = edm_params['sigma_max']
        
        self.edm_sampler = EDMSampler(
            self.diffusion,
            sigma_data=self.sigma_data,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=edm_params['rho'],
            s_churn=edm_params['s_churn'],
            s_noise=edm_params['s_noise']
        )
        
        bert_params = df_conf.bert.params
        self.text_embed_dim = bert_params.n_embed
        self.cond_model = BERTTextEncoder(**bert_params)
        self.cond_model.to(self.device)
        

        trainable_params = []
        trainable_params += [p for p in self.diffusion.parameters() if p.requires_grad]
        trainable_params += [p for p in self.cond_model.parameters() if p.requires_grad]

        if self.isTrain:
            self.edm_loss_fn = EDMLoss(
                sigma_data=self.sigma_data,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max
            )
            
            self.optimizer = optim.AdamW(trainable_params, lr=opt.lr, betas=(0.9, 0.999))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)

        dist, elev, azim = 1.7, 20, 20   
        self.renderer = init_mesh_renderer(
            image_size=256, dist=dist, elev=elev, azim=azim, device=self.device
        )

        if self.opt.distributed:
            self.make_distributed(opt)
            self.diffusion_module = self.diffusion.module
            self.triplane_vae_module = self.triplane_vae.module
            self.cond_model_module = self.cond_model.module
        else:
            self.diffusion_module = self.diffusion
            self.triplane_vae_module = self.triplane_vae
            self.cond_model_module = self.cond_model

        self.num_sampling_steps = 50
        self.cfg_scale = 3.0
        if self.opt.debug == "1":
            self.num_sampling_steps = 10  # faster for debugging
        cprint(f'[*] EDM sampling steps={self.num_sampling_steps}, CFG scale={self.cfg_scale}', 'blue')


    def make_distributed(self, opt):
        self.diffusion = nn.parallel.DistributedDataParallel(
            self.diffusion,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        self.triplane_vae = nn.parallel.DistributedDataParallel(
            self.triplane_vae,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        self.cond_model = nn.parallel.DistributedDataParallel(
            self.cond_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    def set_input(self, input=None, max_sample=None):
        self.x = input['sdf']
        self.text = input['text']
        B = self.x.shape[0]
        self.uc_text = B * [""]

        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.text = self.text[:max_sample]
            self.uc_text = self.uc_text[:max_sample]

        vars_list = ['x']
        self.tocuda(var_names=vars_list)

    def switch_train(self):
        self.diffusion.train()
        self.cond_model.train()
        self.triplane_vae.eval()  # VAE always in eval

    def switch_eval(self):
        self.diffusion.eval()
        self.cond_model.eval()
        self.triplane_vae.eval()

    def forward(self):
        self.switch_train()

        c_text = self.cond_model(self.text)  # B, seq_len, dim

        with torch.no_grad():
            z_0 = self.triplane_vae_module.encode_no_sample(self.x)  # B, 3*z_ch, H, W

        loss = self.edm_loss_fn(self.diffusion, z_0, conditioning=c_text)
        
        self.loss = loss
        self.loss_dict = {
            'loss_total': loss.clone().detach(),
            'loss_diffusion': loss.clone().detach(),
        }

    @torch.no_grad()
    def inference(self, data, num_steps=None, cfg_scale=None, 
                  infer_all=False, max_sample=16):
        self.switch_eval()

        if not infer_all:
            self.set_input(data, max_sample=max_sample)
        else:
            self.set_input(data)

        if num_steps is None:
            num_steps = self.num_sampling_steps
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        c_text = self.cond_model(self.text)
        uc = self.cond_model(self.uc_text)
        B = c_text.shape[0]

        z_gen = self.edm_sampler.sample(
            shape=self.z_shape,
            num_steps=num_steps,
            conditioning=c_text,
            unconditional_conditioning=uc,
            cfg_scale=cfg_scale,
            device=self.device
        )

        self.gen_sdf = self.triplane_vae_module.decode_no_sample(z_gen)

        self.switch_train()

    @torch.no_grad()
    def txt2shape(self, input_txt, ngen=6, num_steps=50, cfg_scale=None):
        self.switch_eval()

        data = {
            'sdf': torch.zeros(ngen),
            'text': [input_txt] * ngen,
        }
        
        self.set_input(data)

        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        c_text = self.cond_model(self.text)
        uc = self.cond_model(self.uc_text)

        z_gen = self.edm_sampler.sample(
            shape=self.z_shape,
            num_steps=num_steps,
            conditioning=c_text,
            unconditional_conditioning=uc,
            cfg_scale=cfg_scale,
            device=self.device
        )

        self.gen_sdf = self.triplane_vae_module.decode_no_sample(z_gen)
        
        return self.gen_sdf

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        
        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])
        
        self.switch_train()
        return ret

    def backward(self):
        self.loss_dict = reduce_loss_dict(self.loss_dict)
        self.loss_total = self.loss_dict['loss_total']
        self.loss_diffusion = self.loss_dict['loss_diffusion']
        
        self.loss.backward()

    def optimize_parameters(self, total_steps):
        self.set_requires_grad([self.diffusion], requires_grad=True)
        self.set_requires_grad([self.cond_model], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        ret = OrderedDict([
            ('total', self.loss_total.mean().data),
            ('diffusion', self.loss_diffusion.mean().data),
        ])
        return ret

    def write_text_on_img(self, text, img_shape=(3, 256, 256)):
        b, c, h, w = len(text), 3, 256, 256
        img_text = np.ones((b, h, w, 3)).astype(np.float32) * 255
        font = cv2.FONT_HERSHEY_COMPLEX
        font_size = 0.5
        n_char_per_line = 25

        y0, dy = 20, 15
        for ix, txt in enumerate(text):
            for i in range(0, len(txt), n_char_per_line):
                y = y0 + (i // n_char_per_line) * dy
                txt_i = txt[i:i+n_char_per_line]
                cv2.putText(img_text[ix], txt_i, (10, y), font, font_size, (0., 0., 0.), 2)

        return img_text / 255.

    def get_current_visuals(self):
        with torch.no_grad():
            self.img_gt = render_sdf(self.renderer, self.x).detach().cpu()
            self.img_gen = render_sdf(self.renderer, self.gen_sdf).detach().cpu()
        
        b, c, h, w = self.img_gt.shape
        self.img_text = self.write_text_on_img(self.text, img_shape=(c, h, w))
        self.img_text = rearrange(torch.from_numpy(self.img_text), 'b h w c -> b c h w')

        vis_tensor_names = ['img_gt', 'img_gen', 'img_text']
        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    def save(self, label, global_step, save_opt=False):
        state_dict = {
            'triplane_vae': self.triplane_vae_module.state_dict(),
            'cond_model': self.cond_model_module.state_dict(),
            'diffusion': self.diffusion_module.state_dict(),
            'edm_params': self.edm_params,
            'global_step': global_step,
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'edm_triplane_df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)
        torch.save(state_dict, save_path)
        print(colored(f'[*] Model saved to: {save_path}', 'blue'))

    def load_vae_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'vae' in state_dict:
            self.triplane_vae.load_state_dict(state_dict['vae'])
        else:
            self.triplane_vae.load_state_dict(state_dict)

    def load_ckpt(self, ckpt, load_opt=False):
        state_dict = torch.load(ckpt, map_location='cpu')

        self.triplane_vae.load_state_dict(state_dict['triplane_vae'])
        self.diffusion.load_state_dict(state_dict['diffusion'])
        self.cond_model.load_state_dict(state_dict['cond_model'])
        
        if 'edm_params' in state_dict:
            self.edm_params = state_dict['edm_params']
        
        print(colored(f'[*] Weight successfully loaded from: {ckpt}', 'blue'))
        
        if load_opt and 'opt' in state_dict:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored(f'[*] Optimizer restored from: {ckpt}', 'blue'))
