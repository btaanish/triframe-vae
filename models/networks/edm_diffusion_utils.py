"""
EDM (Elucidated Diffusion Models) Framework Utilities
Based on "Elucidating the Design Space of Diffusion-Based Generative Models"
https://arxiv.org/abs/2206.00364
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial


class EDMPrecond(nn.Module):
    """
    EDM preconditioning wrapper for the denoising network
    Implements the improved parameterization from EDM paper
    """
    
    def __init__(self, model, sigma_data=0.5, sigma_min=0.002, sigma_max=80.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(self, x, sigma, context=None, **kwargs):
        """
        EDM preconditioning
        Args:
            x: noisy input (B, C, D, H, W)
            sigma: noise level (B,) or (B, 1, 1, 1, 1)
            context: conditioning (B, seq_len, dim)
        Returns:
            denoised output
        """
        # Ensure sigma has correct shape
        if sigma.ndim == 1:
            sigma = sigma.view(-1, 1, 1, 1, 1)
        
        # EDM preconditioning factors
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = torch.log(sigma) / 4  # log sigma encoding
        
        # Precondition input
        x_in = c_in * x
        
        # Get model output with noise conditioning
        # Convert c_noise to timestep-like format for the U-Net
        c_noise_flat = c_noise.squeeze()
        F_x = self.model(x_in, c_noise_flat, context=context, **kwargs)
        
        # Precondition output
        D_x = c_skip * x + c_out * F_x
        
        return D_x


def edm_noise_schedule(n_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """
    EDM's improved noise schedule
    Args:
        n_steps: number of diffusion steps
        sigma_min: minimum noise level
        sigma_max: maximum noise level  
        rho: schedule density parameter (higher = more steps at low noise)
    Returns:
        sigma: noise schedule (n_steps,)
    """
    ramp = torch.linspace(0, 1, n_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def edm_loss_weighting(sigma, sigma_data=0.5):
    """
    EDM's improved loss weighting
    Args:
        sigma: noise level (B,)
        sigma_data: data standard deviation
    Returns:
        weight: loss weight (B,)
    """
    weight = (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
    return weight


class EDMSampler:
    """
    EDM sampler with second-order Heun's method
    """
    
    def __init__(self, model, sigma_data=0.5, sigma_min=0.002, sigma_max=80.0, 
                 rho=7.0, s_churn=0.0, s_noise=1.0):
        """
        Args:
            model: preconditioned denoising model
            sigma_data: data std
            sigma_min: minimum noise
            sigma_max: maximum noise
            rho: noise schedule parameter
            s_churn: stochasticity parameter
            s_noise: noise scale
        """
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.s_churn = s_churn
        self.s_noise = s_noise
    
    @torch.no_grad()
    def sample(self, shape, num_steps=50, conditioning=None, 
               unconditional_conditioning=None, cfg_scale=1.0, device='cuda'):
        """
        Generate samples using EDM's Heun sampler
        Args:
            shape: output shape (C, D, H, W)
            num_steps: number of sampling steps
            conditioning: conditional embedding
            unconditional_conditioning: unconditional embedding for CFG
            cfg_scale: classifier-free guidance scale
            device: torch device
        Returns:
            x: generated samples (B, C, D, H, W)
        """
        B = conditioning.shape[0] if conditioning is not None else 1
        
        # Generate noise schedule
        sigmas = edm_noise_schedule(
            num_steps + 1, 
            self.sigma_min, 
            self.sigma_max, 
            self.rho
        ).to(device)
        
        # Initialize from noise
        x = torch.randn((B, *shape), device=device) * sigmas[0]
        
        # Heun's method sampling
        for i in range(num_steps):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Add noise (stochasticity)
            gamma = min(self.s_churn / num_steps, np.sqrt(2) - 1) if self.s_churn > 0 else 0
            sigma_hat = sigma_cur * (1 + gamma)
            if gamma > 0:
                eps = torch.randn_like(x) * self.s_noise
                x = x + eps * torch.sqrt(sigma_hat ** 2 - sigma_cur ** 2)
            
            # First-order step (Euler)
            denoised = self.denoise_with_cfg(
                x, sigma_hat, conditioning, 
                unconditional_conditioning, cfg_scale
            )
            d_cur = (x - denoised) / sigma_hat
            x_next = x + d_cur * (sigma_next - sigma_hat)
            
            # Second-order correction (Heun)
            if sigma_next > 0:
                denoised_next = self.denoise_with_cfg(
                    x_next, sigma_next, conditioning,
                    unconditional_conditioning, cfg_scale
                )
                d_next = (x_next - denoised_next) / sigma_next
                x_next = x + (d_cur + d_next) * (sigma_next - sigma_hat) / 2
            
            x = x_next
        
        return x
    
    def denoise_with_cfg(self, x, sigma, cond, uncond, scale):
        """
        Denoising with classifier-free guidance
        Args:
            x: noisy input
            sigma: noise level
            cond: conditional embedding
            uncond: unconditional embedding
            scale: guidance scale
        Returns:
            denoised output with CFG applied
        """
        if scale == 1.0 or uncond is None:
            return self.model(x, sigma, context=cond)
        
        # Concatenate for parallel processing
        x_in = torch.cat([x, x], dim=0)
        sigma_in = torch.cat([sigma, sigma], dim=0) if sigma.ndim > 0 else sigma
        context_in = torch.cat([cond, uncond], dim=0)
        
        # Get both predictions
        out = self.model(x_in, sigma_in, context=context_in)
        
        # Split and apply CFG
        cond_out, uncond_out = out.chunk(2, dim=0)
        return uncond_out + scale * (cond_out - uncond_out)


class EDMLoss(nn.Module):
    """
    EDM training loss with improved weighting
    """
    
    def __init__(self, sigma_data=0.5, sigma_min=0.002, sigma_max=80.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(self, model, x_0, conditioning=None):
        """
        Compute EDM training loss
        Args:
            model: preconditioned model (EDMPrecond)
            x_0: clean data (B, C, D, H, W)
            conditioning: conditional embedding
        Returns:
            loss: weighted denoising loss
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample noise levels (log-uniform)
        ln_sigma = torch.randn(B, device=device) * 1.2 - 1.2  # log-normal
        sigma = torch.exp(ln_sigma).clamp(self.sigma_min, self.sigma_max)
        
        # Add noise
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * sigma.view(-1, 1, 1, 1, 1)
        
        # Denoise
        D_x = model(x_t, sigma, context=conditioning)
        
        # Compute loss with EDM weighting
        weight = edm_loss_weighting(sigma, self.sigma_data)
        loss = weight.view(-1, 1, 1, 1, 1) * ((D_x - x_0) ** 2)
        
        return loss.mean()


def create_edm_model(base_model, sigma_data=0.5, sigma_min=0.002, sigma_max=80.0):
    """
    Wrap a base denoising model with EDM preconditioning
    Args:
        base_model: base U-Net or denoising model
        sigma_data: data std
        sigma_min: minimum noise
        sigma_max: maximum noise
    Returns:
        preconditioned model
    """
    return EDMPrecond(base_model, sigma_data, sigma_min, sigma_max)


def get_edm_schedule_params():
    """
    Get recommended EDM parameters
    Returns:
        dict of EDM parameters
    """
    return {
        'sigma_min': 0.002,
        'sigma_max': 80.0,
        'sigma_data': 0.5,
        'rho': 7.0,
        's_churn': 0.0,  # 0 for deterministic, ~40 for stochastic
        's_noise': 1.0,
    }