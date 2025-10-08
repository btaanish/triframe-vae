import os
import h5py
import trimesh
import numpy as np
# import marching_cubes as mcubes
import mcubes
import imageio
import einops
from einops import rearrange, repeat
# from skimage import measure
from termcolor import cprint

import torch
import torchvision.utils as vutils
from skimage import measure

import pytorch3d
import pytorch3d.io
from pytorch3d.structures import Pointclouds, Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)

from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.structures import Meshes

def read_sdf(sdf_h5_file, resolution=64):
    h5_f = h5py.File(sdf_h5_file, 'r')
    sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
    # sdf = torch.Tensor(sdf).view(1, 64, 64, 64)
    sdf = torch.Tensor(sdf).view(1, resolution, resolution, resolution)
    sdf = sdf[None, ...]
    return sdf

def save_mesh_as_gif(mesh_renderer, mesh, nrow=3, fps=450, out_name='1.gif'):

    rot_comb = rotate_mesh_360(mesh_renderer, mesh)
    
    nimgs = len(rot_comb)
    nrots = len(rot_comb[0])
    H, W, C = rot_comb[0][0].shape
    rot_comb_img = []
    for i in range(nrots):
        img_grid_i = torch.zeros(nimgs, H, W, C)
        for j in range(nimgs):
            img_grid_i[j] = torch.from_numpy(rot_comb[j][i])
            img_grid_i[j, :, :, 3] = 255
            
        img_grid_i = img_grid_i.permute(0, 3, 1, 2)
        img_grid_i = vutils.make_grid(img_grid_i, nrow=nrow)
        img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)
            
        rot_comb_img.append(img_grid_i)
    

    duration = nrots / fps

    with imageio.get_writer(out_name, mode='I', duration=duration) as writer:
        
        for rot in rot_comb_img:
            writer.append_data(rot)


def init_points_renderer(image_size=256, dist=1.7, elev=20, azim=20, camera='0', device='cuda:0'):
    R, T = look_at_view_transform(dist, elev, azim) 

    if camera == '0':
        camera_cls = FoVPerspectiveCameras
    else:
        camera_cls = FoVOrthographicCameras

    cameras = camera_cls(device=device, R=R, T=T)

    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        # radius = 0.0,
        radius=0.003,
        # points_per_pixel = 10
        # bin_size=None,
        points_per_pixel = 1
    )


    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    return renderer

def init_mesh_renderer(image_size=512, dist=3.5, elev=90, azim=90, camera='0', device='cuda:0'):

    if camera == '0':
        camera_cls = FoVPerspectiveCameras
    else:
        camera_cls = FoVOrthographicCameras
    
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = camera_cls(device=device, R=R, T=T)
    # print(f'[*] renderer device: {device}')

    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0, 
        faces_per_pixel=1, 
    )

    lights = PointLights(device=device, location=[[1.0, 1.0, 0.0]])

    # apply the Phong lighting model
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras, 
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftPhongShader(
    #         device=device, 
    #         cameras=cameras,
    #         lights=lights
    #     )
    # )
    renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras)
        )
    return renderer


def marching_cubes_compat(vol, level):
    import numpy as np
    import torch
    try:
        import mcubes
    except ImportError:
        mcubes = None

    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()
    vol = np.asarray(vol, dtype=np.float32, order="C")
    level = float(level)

    if mcubes is not None and hasattr(mcubes, "marching_cubes"):
        try:
            return mcubes.marching_cubes(vol, level)
        except Exception:
            pass

    if mcubes is not None and hasattr(mcubes, "MarchingCubes"):
        dims = tuple(int(x) for x in vol.shape)  # MUST be a plain tuple
        for kwargs in (
            dict(dims=dims, sampling_interval=1, offset=(0, 0, 0), values=None, iso_level=level),
            dict(size=dims, sampling_interval=1, offset=(0, 0, 0), values=None, iso_level=level),
        ):
            try:
                mc = mcubes.MarchingCubes(vol, **kwargs)
                return mc.vertices, mc.triangles
            except Exception:
                pass
        try:
            mc = mcubes.MarchingCubes(vol, dims, 1, (0, 0, 0), None, level)
            return mc.vertices, mc.triangles
        except Exception:
            pass

    verts, faces, _normals, _values = measure.marching_cubes(vol, level=level)
    return verts.astype(np.float32), faces.astype(np.int32)

def sdf_to_mesh(sdf, level=0.02, color=None, render_all=False):
    device=sdf.device

    n_cell = sdf.shape[-1]
    bs, nc = sdf.shape[:2]

    assert nc == 1

    nimg_to_render = bs
    if not render_all:
        if bs > 16:
            cprint('Warning! Will not return all meshes', 'red')
        nimg_to_render = min(bs, 16)

    verts = []
    faces = []
    verts_rgb = []

    for i in range(nimg_to_render):
        sdf_i = sdf[i, 0].detach().cpu().numpy()
        if isinstance(sdf_i, torch.Tensor):
            sdf_i = sdf_i.detach().cpu().numpy()

        dims = tuple(int(d) for d in sdf_i.shape)

        level = float(level)

        verts_i, faces_i = marching_cubes_compat(sdf_i, level)
        verts_i = verts_i / n_cell - .5 

        verts_i = torch.from_numpy(verts_i).float().to(device)
        faces_i = torch.from_numpy(faces_i.astype(np.int64)).to(device)
        text_i = torch.ones_like(verts_i).to(device)
        if color is not None:
            for i in range(3):
                text_i[:, i] = color[i]

        verts.append(verts_i)
        faces.append(faces_i)
        verts_rgb.append(text_i)

    try:
        p3d_mesh = pytorch3d.structures.Meshes(verts, faces, textures=pytorch3d.renderer.Textures(verts_rgb=verts_rgb))
    except:
        p3d_mesh = None

    return p3d_mesh

def voxel_to_mesh(voxel, color=None):
    vox_mesh = pytorch3d.ops.cubify(voxel, thresh=0.5)
    verts = vox_mesh.verts_list()
    verts_rgb_list = []
    for i in range(len(verts)):
        verts_rgb = torch.ones_like(verts[i])
        if color is not None:
            for i in range(3):
                verts_rgb[:, i] = color[i]
        verts_rgb_list.append(verts_rgb)

    vox_mesh.textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
    return vox_mesh

def render_pcd(renderer, verts, color=[1, 1, 1], alpha=False):
    if verts.dim() == 2:
        verts = verts[None, ...]

    verts = verts.to(renderer.rasterizer.cameras.device)
    # verts = verts.cpu()

    V = verts.shape[1]
    B = verts.shape[0]
    features=torch.ones_like(verts)
    for i in range(3):
        features[:, :, i] = color[i]
    pcl = Pointclouds(points=verts, features=features)
    try:
        images = renderer(pcl)
    except:
        images = renderer(pcl, gamma=(1e-4,),)
        
    return images.permute(0, 3, 1, 2)

def render_mesh(renderer, mesh, color=None, norm=True):
    if mesh.textures is None:
        verts = mesh.verts_list()
        verts_rgb_list = []
        for i in range(len(verts)):
        # print(verts.min(), verts.max())
            verts_rgb_i = torch.ones_like(verts[i])
            if color is not None:
                for i in range(3):
                    verts_rgb_i[:, i] = color[i]
            verts_rgb_list.append(verts_rgb_i)

        texture = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
        mesh.textures = texture

    images = renderer(mesh)
    return images.permute(0, 3, 1, 2)

def render_voxel(mesh_renderer, voxel, render_all=False):

    bs = voxel.shape[0]
    if not render_all:
        nimg_to_render = min(bs, 16)
        voxel = voxel[:nimg_to_render]
    else:
        nimg_to_render = bs

    meshes = pytorch3d.ops.cubify(voxel, thresh=0.5)
    # verts = meshes.verts_list()[0][None, ...]
    verts_list = meshes.verts_list()
    norm_verts_list = []
    verts_rgb_list = []
    for verts in verts_list:
    # print(f'verts: {verts.min()}, {verts.max()}, {verts.mean()}')
        try:
            verts = (verts - verts.min()) / (verts.max() - verts.min())
        except:
            images = torch.zeros(nimg_to_render, 4, 256, 256).to(voxel)
            return images

        verts = verts * 2 - 1
        norm_verts_list.append(verts)
        verts_rgb_list.append(torch.ones_like(verts))

    meshes.textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
    try:
        images = mesh_renderer(meshes)
        images = images.permute(0, 3, 1, 2)
    except:
        images = torch.zeros(nimg_to_render, 4, 256, 256).to(voxel)
        print('here')

    return images


def add_mesh_textures(mesh):
    verts = mesh.verts_list()
    faces = mesh.faces_list()
    
    bs = len(verts)
    verts_rgb = []
    for i in range(bs):
        verts_rgb.append(torch.ones_like(verts[i]))
    
    # mesh = Meshes(verts=verts, faces=faces, textures=pytorch3d.renderer.mesh.TexturesVertex(verts_features=verts_rgb))
    mesh.textures = pytorch3d.renderer.mesh.TexturesVertex(verts_rgb)
    return mesh

def render_sdf(mesh_renderer, sdf, level=0.02, color=None, render_imsize=256, render_all=False):
    device = sdf.device
    bs = sdf.shape[0]

    if not render_all:
        nimg_to_render = min(bs, 16)
    
    p3d_mesh = sdf_to_mesh(sdf, level=level, color=color, render_all=render_all)

    if p3d_mesh is not None:
        rendered_im = einops.rearrange(mesh_renderer(p3d_mesh), 'b h w c-> b c h w').contiguous()
    else:
        rendered_im = torch.zeros(nimg_to_render, 4, render_imsize, render_imsize).to(device)

    return rendered_im


def rotate_mesh(mesh, axis='Y', angle=10, device='cuda'):
    rot_func = RotateAxisAngle(angle, axis, device=device)

    verts = mesh.verts_list()
    faces = mesh.faces_list()
    textures = mesh.textures
    
    B = len(verts)

    rot_verts = []
    for i in range(B):
        v = rot_func.transform_points(verts[i])
        rot_verts.append(v)
    new_mesh = Meshes(verts=rot_verts, faces=faces, textures=textures).to(device)
    return new_mesh

def rotate_mesh_360(mesh_renderer, mesh, n_frames=36):
    device = mesh.device
    cur_mesh = mesh

    B = len(mesh.verts_list())
    ret = [ [] for i in range(B)]

    angle = (360 // n_frames)

    # for i in range(36):
    for i in range(n_frames):
        cur_mesh = rotate_mesh(cur_mesh, angle=angle, device=device)
        img = render_mesh(mesh_renderer, cur_mesh, norm=False) # b c h w
        img = img.permute(0, 2, 3, 1) # b h w c
        img = img.detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        for j in range(B):
            ret[j].append(img[j])

    return ret



def combine_meshes(mesh1, mesh2):
    if mesh1 is None:
        return mesh2
    if mesh2 is None:
        return mesh1
    
    comb_mesh_list = []
    for ix, (m1, m2) in enumerate(zip(mesh1, mesh2)):
        comb_mesh = pytorch3d.structures.join_meshes_as_scene([m1, m2])
        comb_mesh_list.append(comb_mesh)
    comb_mesh = pytorch3d.structures.join_meshes_as_batch(comb_mesh_list)
    return comb_mesh



def load_mesh(obj_f):
    verts, faces_tup, _ = pytorch3d.io.load_obj(obj_f, load_textures=False)
    faces = faces_tup.verts_idx

    verts = verts.unsqueeze(0)
    faces = faces.unsqueeze(0)

    verts_rgb = torch.ones_like(verts)
    mesh = pytorch3d.structures.Meshes(verts=verts, faces=faces, textures=pytorch3d.renderer.TexturesVertex(verts_rgb))

    return mesh

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = trimesh.Trimesh(vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces)
    return mesh

def get_normalize_mesh(model_file):
    total = 16384
    # print("[*] trimesh_load:", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)

    mesh = as_mesh(mesh_list) # from s2s
    if not isinstance(mesh, list):
        mesh_list = [mesh]

    area_sum = 0
    area_lst = []
    for idx, mesh in enumerate(mesh_list):
        area = np.sum(mesh.area_faces)
        area_lst.append(area)
        area_sum+=area
    area_lst = np.asarray(area_lst)
    amount_lst = (area_lst * total / area_sum).astype(np.int32)
    points_all=np.zeros((0,3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        # print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    # obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    ori_mesh_list = trimesh.load_mesh(model_file, process=False)
    ori_mesh = as_mesh(ori_mesh_list)
    ori_mesh.vertices = (ori_mesh.vertices - centroid) / float(m)
    return ori_mesh, centroid, m

