import os
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader
)
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the OBJ file
mesh = load_objs_as_meshes(['/root/autodl-tmp/SDFusion/test_results_pretrained_small_dset/obj/txt2shape-articulated zigzag fold-based 1dof directional bending soft actuator-35.obj'], device=device)

# Calculate the centroid of the mesh
centroid = mesh.verts_packed().mean(0)

# Translate the mesh to center it at the origin
translate = -centroid
mesh.offset_verts_(translate)

# Camera settings: orthographic, looking at the origin from different views
camera_positions = {
    'top': (0, 0, 1),
    'bottom': (0, 0, -1),
    'front': (0, 1, 0),
    'back': (0, -1, 0),
    'left': (-1, 0, 0),
    'right': (1, 0, 0)
}

# Common setup for all views
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1,
)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=OpenGLPerspectiveCameras(device=device),
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader()
)

# Render all views and save
output_dir = '/root/autodl-tmp/rendered_imgs'
os.makedirs(output_dir, exist_ok=True)

for view_name, pos in camera_positions.items():
    R, T = look_at_view_transform(dist=4.0, elev=0, azim=0, at=((0, 0, 0),), up=((0, 1, 0),), device=device)
    T += torch.tensor(pos, dtype=torch.float32, device=device).unsqueeze(0)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    images = renderer(mesh.extend(1), cameras=cameras)
    image_np = images[0, ..., 3].cpu().numpy()  # Get the alpha channel which contains the silhouette
    plt.imsave(os.path.join(output_dir, f'{view_name}_view.png'), image_np, cmap='gray')

print("All views have been rendered using PyTorch3D.")
