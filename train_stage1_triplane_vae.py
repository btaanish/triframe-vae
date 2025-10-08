import os
import sys
from options.train_options import TrainOptions
from datasets.dataloader import CreateDataLoader
from models.base_model import create_model
from utils.visualizer import Visualizer
from train import train_main_worker

if __name__ == "__main__":
    # Parse options
    opt = TrainOptions().parse_and_setup()
    
    # Override with Stage 1 settings
    opt.model = 'triplane-vae'
    opt.vq_cfg = 'configs/triplane_vae_config.yaml'
    
    print(f"[*] Starting Triplane VAE Training")
    print(f"    Model: {opt.model}")
    print(f"    Config: {opt.vq_cfg}")
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
