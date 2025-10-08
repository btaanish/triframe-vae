import os
import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Force Pyrender to use OSMesa or EGL
pyrender.OFFSCREEN_RENDERING = True  # This is critical for headless setups

# Ensure EGL is used for offscreen rendering
if os.environ.get('PYOPENGL_PLATFORM') is None:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Load the OBJ file
mesh = trimesh.load('/data/Softnet/01/a5/model.obj')

scene = pyrender.Scene()
pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
scene.add(pyrender_mesh)

# Setup camera and light as before
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
s = np.sqrt(2)/2
camera_pose = np.array([
   [1, 0,  0, 0],
   [0, 1,  0, 0],
   [0, 0,  1, 2],
   [0, 0,  0, 1]
])

light = pyrender.SpotLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)

# Use OSMesaRenderer or EGLRenderer based on your setup
renderer = pyrender.OffscreenRenderer(512, 512, point_size=1.0)

camera_poses = []
camera_poses.append(np.array([
    [1, 0,  0, 0],
    [0, 1,  0, 0],
    [0, 0,  1, 1],
    [0, 0,  0, 1]
]))
# next, rotate the camera 90 degrees around the x-axis
camera_poses.append(np.array([
    [1, 0,  0, 0],
    [0, 0, -1, -1],
    [0, 1,  0, 0],
    [0, 0,  0, 1]
]))
# next, rotate the camera 90 degrees around the x-axis
camera_poses.append(np.array([
    [1, 0,  0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, -1],
    [0, 0,  0, 1]
]))
# next, rotate the camera 90 degrees around the x-axis
camera_poses.append(np.array([
    [1, 0,  0, 0],
    [0, 0, 1, 1],
    [0, -1,  0, 0],
    [0, 0,  0, 1]
]))
# next, rotate the camera 90 degrees around the y-axis, top view
camera_poses.append(np.array([
    [0, 0, 1, 1],
    [0, 1,  0, 0],
    [-1, 0,  0, 0],
    [0, 0,  0, 1]
]))
# finally, rotate the camera -90 degrees around the y-axis, bottom view
camera_poses.append(np.array([
    [0, 0, -1, -1],
    [0, 1,  0, 0],
    [1, 0,  0, 0],
    [0, 0,  0, 1]
]))


for i, camera_pose in enumerate(camera_poses):
    scene.add(camera, pose=camera_pose)
    color, depth = renderer.render(scene)
    scene.remove_node(scene.get_nodes(obj=camera).pop())

    # Save the rendered image
    image_path = f'/root/autodl-tmp/rendered_imgs/view_{i}.png'
    plt.imsave(f'{image_path}', color)

    print(f'Rendered {image_path}')

renderer.delete()
print("Rendering complete.")
