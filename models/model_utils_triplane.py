"""
Model utilities for loading Triplane VAE
"""

import torch
from omegaconf import OmegaConf
from termcolor import colored

from models.networks.triplane_vae_network import TriplaneVAE


def load_triplane_vae(vq_conf, vq_ckpt=None, opt=None):
    """
    Load Triplane VAE model
    
    Args:
        vq_conf: OmegaConf config object or path to config
        vq_ckpt: path to checkpoint (optional)
        opt: training options (optional)
    
    Returns:
        triplane_vae: loaded TriplaneVAE model
    """
    # Load config if it's a path
    if isinstance(vq_conf, str):
        vq_conf = OmegaConf.load(vq_conf)
    
    # Extract parameters
    mparam = vq_conf.model.params
    ddconfig = mparam.ddconfig
    
    resolution = ddconfig.resolution
    z_channels = ddconfig.z_channels
    base_channels = ddconfig.ch
    n_downsamples = len(ddconfig.ch_mult) - 1
    
    # Create model
    triplane_vae = TriplaneVAE(
        in_channels=1,
        z_channels=z_channels,
        resolution=resolution,
        base_channels=base_channels,
        hidden_dim=256,
        n_downsamples=n_downsamples
    )
    
    # Load checkpoint if provided
    if vq_ckpt is not None:
        state_dict = torch.load(vq_ckpt, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'vae' in state_dict:
            triplane_vae.load_state_dict(state_dict['vae'])
        else:
            triplane_vae.load_state_dict(state_dict)
        
        print(colored(f'[*] Triplane VAE loaded from: {vq_ckpt}', 'green'))
    
    # Move to device if opt is provided
    if opt is not None and hasattr(opt, 'device'):
        triplane_vae = triplane_vae.to(opt.device)
    
    return triplane_vae


def load_edm_text2shape_model(df_conf, vq_conf, vq_ckpt=None, df_ckpt=None, opt=None):
    """
    Load complete EDM text-to-shape model
    
    Args:
        df_conf: diffusion config (OmegaConf or path)
        vq_conf: VAE config (OmegaConf or path)
        vq_ckpt: VAE checkpoint path (optional)
        df_ckpt: diffusion checkpoint path (optional)
        opt: training options (optional)
    
    Returns:
        model: EDMTriplaneText2ShapeModel
    """
    from models.edm_triplane_text2shape_model import EDMTriplaneText2ShapeModel
    
    # Create model
    if opt is None:
        # Create minimal opt for inference
        class MinimalOpt:
            def __init__(self):
                self.isTrain = False
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.distributed = False
                self.debug = "0"
                self.vq_cfg = vq_conf if isinstance(vq_conf, str) else None
                self.df_cfg = df_conf if isinstance(df_conf, str) else None
                self.vq_ckpt = vq_ckpt
                self.ckpt = df_ckpt
                self.dataset_mode = 'shapenet_lang'
        
        opt = MinimalOpt()
    
    model = EDMTriplaneText2ShapeModel()
    model.initialize(opt)
    
    return model


def get_triplane_latent_shape(vq_conf):
    """
    Get the shape of triplane latent from config
    
    Args:
        vq_conf: VAE config (OmegaConf or path)
    
    Returns:
        z_shape: (3*z_channels, triplane_dim, triplane_dim)
    """
    if isinstance(vq_conf, str):
        vq_conf = OmegaConf.load(vq_conf)
    
    mparam = vq_conf.model.params
    ddconfig = mparam.ddconfig
    
    resolution = ddconfig.resolution
    z_channels = ddconfig.z_channels
    n_downsamples = len(ddconfig.ch_mult) - 1
    
    triplane_dim = resolution // (2 ** n_downsamples)
    z_shape = (3 * z_channels, triplane_dim, triplane_dim)
    
    return z_shape


def test_triplane_vae(vae, device='cuda'):
    """
    Test triplane VAE with random input
    
    Args:
        vae: TriplaneVAE model
        device: torch device
    """
    vae.eval()
    
    # Create random SDF input
    x = torch.randn(2, 1, 64, 64, 64).to(device)
    
    print("Testing Triplane VAE...")
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        # Encode
        mu, logvar = vae.encode(x)
        print(f"Latent mu shape: {mu.shape}")
        print(f"Latent logvar shape: {logvar.shape}")
        
        # Decode
        x_recon, _, _ = vae(x)
        print(f"Reconstruction shape: {x_recon.shape}")
        
        # Check reconstruction quality
        mse = torch.nn.functional.mse_loss(x, x_recon)
        print(f"Reconstruction MSE: {mse.item():.6f}")
    
    print("Triplane VAE test passed!")


if __name__ == "__main__":
    # Example usage
    print("Testing Triplane VAE loading...")
    
    vq_conf_path = "configs/triplane_vae_config.yaml"
    
    # Load VAE
    vae = load_triplane_vae(vq_conf_path)
    
    # Test it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = vae.to(device)
    test_triplane_vae(vae, device)
    
    print("\nAll tests passed!")