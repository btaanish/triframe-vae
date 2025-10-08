import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.functional as F

from models.networks.vqvae_networks.vqvae_modules import Encoder3D, Decoder3D


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'init method {init_type} not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)


class TriplaneEncoder(nn.Module):
    
    def __init__(self, in_channels=1, base_channels=64, z_channels=32, 
                 resolution=64, n_downsamples=3):
        super().__init__()
        
        self.resolution = resolution
        self.z_channels = z_channels
        self.n_downsamples = n_downsamples
        
        self.encoder_3d = nn.ModuleList()
        curr_ch = base_channels
        
        self.encoder_3d.append(nn.Conv3d(in_channels, curr_ch, 3, padding=1))
        self.encoder_3d.append(nn.GroupNorm(8, curr_ch))
        self.encoder_3d.append(nn.SiLU())
        
        for i in range(n_downsamples):
            next_ch = curr_ch * 2
            self.encoder_3d.append(nn.Conv3d(curr_ch, next_ch, 3, stride=2, padding=1))
            self.encoder_3d.append(nn.GroupNorm(min(32, next_ch // 4), next_ch))
            self.encoder_3d.append(nn.SiLU())
            curr_ch = next_ch
        
        self.triplane_dim = resolution // (2 ** n_downsamples)
        
        self.to_xy_plane = nn.Sequential(
            nn.Conv3d(curr_ch, z_channels, 1),
            nn.GroupNorm(min(8, z_channels // 4), z_channels),
            nn.SiLU()
        )
        self.to_xz_plane = nn.Sequential(
            nn.Conv3d(curr_ch, z_channels, 1),
            nn.GroupNorm(min(8, z_channels // 4), z_channels),
            nn.SiLU()
        )
        self.to_yz_plane = nn.Sequential(
            nn.Conv3d(curr_ch, z_channels, 1),
            nn.GroupNorm(min(8, z_channels // 4), z_channels),
            nn.SiLU()
        )
        
        self.latent_channels = 3 * z_channels
        self.fc_mu     = nn.Conv2d(self.latent_channels, self.latent_channels, 1)
        self.fc_logvar = nn.Conv2d(self.latent_channels, self.latent_channels, 1) 
    def forward(self, x):
        h = x
        for layer in self.encoder_3d:
            h = layer(h)
        
        B, C, D, H, W = h.shape
        
        xy_feat = torch.mean(h, dim=2)
        xz_feat = torch.mean(h, dim=3)
        yz_feat = torch.mean(h, dim=4)
        
        xy_plane = self.to_xy_plane(h.mean(dim=2, keepdim=True)).squeeze(2)
        xz_plane = self.to_xz_plane(h.mean(dim=3, keepdim=True)).squeeze(3)
        yz_plane = self.to_yz_plane(h.mean(dim=4, keepdim=True)).squeeze(4)
        
        triplane = torch.cat([xy_plane, xz_plane, yz_plane], dim=1)
        
        mu = self.fc_mu(triplane)
        logvar = self.fc_logvar(triplane)
        
        return triplane, mu, logvar


class TriplaneDecoder(nn.Module):
    
    def __init__(self, z_channels=32, hidden_dim=256, out_channels=1, 
                 resolution=64, triplane_dim=8):
        super().__init__()
        
        self.resolution = resolution
        self.z_channels = z_channels
        self.triplane_dim = triplane_dim
        
        self.sdf_mlp = nn.Sequential(
            nn.Linear(3 * z_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_channels)
        )
        
        for m in self.sdf_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def sample_plane_features(self, plane, coords):
        coords = coords.unsqueeze(2)  
        features = F.grid_sample(
            plane, coords, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        return features.squeeze(-1)  
    
    def forward(self, triplane):
        B = triplane.shape[0]
        
        xy_plane = triplane[:, :self.z_channels]
        xz_plane = triplane[:, self.z_channels:2*self.z_channels]
        yz_plane = triplane[:, 2*self.z_channels:]
        
        coords = torch.linspace(-1, 1, self.resolution, device=triplane.device)
        grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
        
        grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        grid_xyz = grid_xyz.unsqueeze(0).repeat(B, 1, 1)
        
        xy_coords = grid_xyz[..., [0, 1]]
        xz_coords = grid_xyz[..., [0, 2]]
        yz_coords = grid_xyz[..., [1, 2]] 
        
        xy_feats = self.sample_plane_features(xy_plane, xy_coords)
        xz_feats = self.sample_plane_features(xz_plane, xz_coords)
        yz_feats = self.sample_plane_features(yz_plane, yz_coords)
        
        combined_feats = torch.cat([xy_feats, xz_feats, yz_feats], dim=1)
        
        combined_feats = combined_feats.permute(0, 2, 1)
        
        sdf = self.sdf_mlp(combined_feats)
        
        sdf = sdf.reshape(B, self.resolution, self.resolution, self.resolution, 1)
        sdf = sdf.permute(0, 4, 1, 2, 3)
        
        return sdf


class TriplaneVAE(nn.Module):
    
    def __init__(self, in_channels=1, z_channels=32, resolution=64, 
                 base_channels=64, hidden_dim=256, n_downsamples=3):
        super().__init__()
        
        self.resolution = resolution
        self.z_channels = z_channels
        self.triplane_dim = resolution // (2 ** n_downsamples)
        
        self.encoder = TriplaneEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            z_channels=z_channels,
            resolution=resolution,
            n_downsamples=n_downsamples
        )
        
        self.decoder = TriplaneDecoder(
            z_channels=z_channels,
            hidden_dim=hidden_dim,
            out_channels=in_channels,
            resolution=resolution,
            triplane_dim=self.triplane_dim
        )
        
        init_weights(self.encoder, 'kaiming', 0.02)
        init_weights(self.decoder, 'kaiming', 0.02)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        triplane, mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z):
        sdf = self.decoder(z)
        return sdf
    
    def forward(self, x, return_latent=False):
        _, mu, logvar = self.encoder(x)
        
        z = self.reparameterize(mu, logvar)
        
        recon = self.decode(z)
        
        if return_latent:
            return recon, mu, logvar, z
        return recon, mu, logvar
    
    def encode_no_sample(self, x):
        _, mu, _ = self.encoder(x)
        return mu
    
    def decode_no_sample(self, z):
        return self.decode(z)
