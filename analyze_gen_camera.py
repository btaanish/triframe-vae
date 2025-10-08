import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Load the OBJ file
# mesh = trimesh.load('/root/autodl-tmp/SDFusion/data/ShapeNet/SoftNetModels/01/a3/model.obj')
mesh = trimesh.load('/root/autodl-tmp/SDFusion/test_results_pretrained_small_dset/obj/txt2shape-articulated zigzag fold-based 1dof directional bending soft actuator-35.obj')

# Ensure the output directory exists
output_dir = '/root/autodl-tmp/rendered_imgs'
os.makedirs(output_dir, exist_ok=True)

# Function to plot views
def plot_view(ax, vertices, faces_sorted, coord_indices, color_indices):
    for face in faces_sorted:
        face = np.append(face, face[0])  # Close the loop
        ax.fill(vertices[face, coord_indices[0]], vertices[face, coord_indices[1]], 
                color=plt.cm.jet((vertices[face, color_indices].mean() - vertices[:, color_indices].min()) / 
                                 (vertices[:, color_indices].max() - vertices[:, color_indices].min())))

# Get faces and vertices
faces = mesh.faces
vertices = mesh.vertices

# Axes to use for each view: (x, y, z) -> index (0, 1, 2)
views = {
    'top': (0, 1, 2),
    'bottom': (0, 1, 2),
    'front': (0, 2, 1),
    'back': (0, 2, 1),
    'left': (1, 2, 0),
    'right': (1, 2, 0)
}

# Sorting axis for each view
sorting_axes = {
    'top': 2,
    'bottom': 2,
    'front': 1,
    'back': 1,
    'left': 0,
    'right': 0
}

# Flip sorting order for these views
reverse_sort = ['bottom', 'back', 'right']

for view_name, axes in views.items():
    fig, ax = plt.subplots()
    # Determine sorting axis and order
    sort_axis = sorting_axes[view_name]
    sorted_faces = faces[np.argsort(vertices[faces].mean(axis=1)[:, sort_axis])]

    if view_name in reverse_sort:
        sorted_faces = sorted_faces[::-1]
    
    plot_view(ax, vertices, sorted_faces, (axes[0], axes[1]), axes[2])
    ax.set_aspect('equal')
    plt.savefig(os.path.join(output_dir, f'{view_name}_view.png'))
    plt.close(fig)

print("All views have been rendered.")
