import os
from options.train_options import TrainOptions
from datasets.dataloader import CreateDataLoader
from models.base_model import create_model
from utils.visualizer import Visualizer
from train import train_main_worker

if __name__ == "__main__":
    opt = TrainOptions().parse_and_setup()
    
    # Override with Stage 2 settings
    opt.model = 'edm-triplane-txt2shape'
    opt.vq_cfg = 'configs/triplane_vae_config.yaml'
    opt.df_cfg = 'configs/edm_diffusion_config.yaml'
    
    # IMPORTANT: Point to trained VAE from Stage 1
    opt.vq_ckpt = '/logs/continue-triplane_vae_softnet/ckpt/triplane_vae_steps-latest.pth'
    
    print(f"[*] Starting EDM Diffusion Training")
    print(f"    Model: {opt.model}")
    print(f"    VAE checkpoint: {opt.vq_ckpt}")
    print(f"    Device: {opt.device}")
    
    # Create dataloaders
    train_dl, test_dl, test_dl_for_eval = CreateDataLoader(opt)
    
    # Create model
    model = create_model(opt)
    
    # Create visualizer
    visualizer = Visualizer(opt)
    visualizer.setup_io()
    
    # Train
    train_main_worker(opt, model, train_dl, test_dl, test_dl_for_eval, visualizer, opt.device)
