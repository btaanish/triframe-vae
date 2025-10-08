import os
import torch
from termcolor import cprint
from models.base_model import create_model
from utils.util_3d import sdf_to_mesh, save_mesh_as_gif, rotate_mesh
from pytorch3d.io import save_obj
class InferenceOpt:
def __init__(self, gpu_id=0):
self.gpu_ids = [gpu_id]
self.device = torch.device(f'cuda:{gpu_id}')
self.isTrain = False
self.distributed = False
self.debug = "0"
self.dataset_mode = 'shapenet_lang'
self.model = 'edm-triplane-txt2shape'
self.vq_cfg = 'configs/triplane_vae_config.yaml'
self.df_cfg = 'configs/edm_diffusion_config.yaml'
self.vq_ckpt = 'logs/triplane_vae_softnet/ckpt/triplane_vae_epoch-best.pth'
self.ckpt = 'logs/edm_diffusion_softnet/ckpt/edm_triplane_df_steps-latest.pth'
def generate_shapes(text_prompts, output_dir='test_results_edm'):
opt = InferenceOpt(gpu_id=0)
model = create_model(opt)
cprint(f'[*] "{model.name()}" loaded.', 'cyan')
os.makedirs(f'{output_dir}/gif', exist_ok=True)
os.makedirs(f'{output_dir}/obj', exist_ok=True)
for i, text in enumerate(text_prompts):
print(f'[{i+1}/{len(text_prompts)}] Generating: "{text}"')
sdf_gen = model.txt2shape(
input_txt=text,
ngen=1,
num_steps=50,
cfg_scale=3.0
)
mesh_gen = sdf_to_mesh(sdf_gen)
gif_path = f'{output_dir}/gif/{i:03d}-{text[:50]}.gif'
save_mesh_as_gif(model.renderer, mesh_gen, nrow=1, out_name=gif_path)
obj_path = f'{output_dir}/obj/{i:03d}-{text[:50]}.obj'
save_obj(obj_path, mesh_gen.verts_list()[0], mesh_gen.faces_list()[0])
print(f' Saved: {gif_path}')
print(f' Saved: {obj_path}')
if __name__ == "__main__":
test_prompts = [
'fold based 1 dof pure bending soft actuator',
'long bellow type 1 dof pure bending soft actuator',
'dome shaped cylindrical fold based normal bidirectional 2dof bending soft actuator',
]
generate_shapes(test_prompts)
