import os
import h5py
import numpy as np
import torch
from datasets.base_dataset import BaseDataset

class SoftnetSDFDataset(BaseDataset):
    def name(self):
        return 'SoftnetSDFDataset'
    
    def initialize(self, opt, phase='train', cat='all', res=64):
        self.opt = opt
        self.phase = phase
        self.res = res
        
        # Get all SDF files
        self.dataroot = opt.dataroot
        self.sdf_files = []
        
        print(f"[*] Loading Softnet SDF data from: {self.dataroot}")
        
        # Find all .h5 files
        for item in os.listdir(self.dataroot):
            item_path = os.path.join(self.dataroot, item)
            if os.path.isdir(item_path):
                h5_file = os.path.join(item_path, 'ori_sample_grid.h5')
                if os.path.exists(h5_file):
                    self.sdf_files.append(h5_file)
        
        # Split train/test (80/20)
        num_samples = len(self.sdf_files)
        split_idx = int(num_samples * 0.8)
        
        if phase == 'train':
            self.sdf_files = self.sdf_files[:split_idx]
        else:
            self.sdf_files = self.sdf_files[split_idx:]
        
        print(f"[*] Found {len(self.sdf_files)} {phase} samples")
    
    def __len__(self):
        return len(self.sdf_files)
    
    def __getitem__(self, index):
        # Load SDF
        sdf_path = self.sdf_files[index]
        
        with h5py.File(sdf_path, 'r') as f:
            sdf = f['pc_sdf_sample'][:].astype(np.float32)
        
        # Reshape to (1, res, res, res)
        sdf = torch.FloatTensor(sdf).view(1, self.res, self.res, self.res)
        
        return {
            'sdf': sdf,
            'path': sdf_path
        }
