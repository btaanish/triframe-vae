import numpy as np

def read_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                if parts[0] == 'v':
                    vertices.append(list(map(float, parts[1:4])))
                elif parts[0] == 'f':
                    faces.append([int(p.split('/')[0]) - 1 for p in parts[1:]])

    return np.array(vertices), faces

def calculate_normals(vertices, faces):
    normals = []
    for face in faces:
        v1, v2, v3 = [vertices[idx] for idx in face]
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        normals.append(normal / np.linalg.norm(normal))

    return np.array(normals)

def calculate_vertex_normals(vertices, faces, face_normals):
    vertex_normals = [np.zeros(3) for _ in vertices]
    face_count = [0] * len(vertices)

    for i, face in enumerate(faces):
        for v in face:
            vertex_normals[v] += face_normals[i]
            face_count[v] += 1

    for i in range(len(vertex_normals)):
        vertex_normals[i] /= face_count[i]
        vertex_normals[i] /= np.linalg.norm(vertex_normals[i])

    return vertex_normals

def normal_variation(normals, faces):
    angle_diffs = []

    # Find adjacent faces
    from collections import defaultdict
    edge_faces = defaultdict(list)
    for i, face in enumerate(faces):
        for j in range(len(face)):
            edge = tuple(sorted((face[j], face[(j + 1) % len(face)])))
            edge_faces[edge].append(i)

    # Calculate angles between normals of adjacent faces
    for adj_faces in edge_faces.values():
        for i in range(len(adj_faces)):
            for j in range(i + 1, len(adj_faces)):
                normal1 = normals[adj_faces[i]]
                normal2 = normals[adj_faces[j]]
                cosine_angle = np.dot(normal1, normal2)
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                angle = np.arccos(cosine_angle)
                angle_diffs.append(angle)
    
    return np.std(angle_diffs), np.max(angle_diffs)

import os

max_variations = []
std_deviations = []

# gt
# obj_dir = "/root/autodl-tmp/SDFusion/data/ShapeNet/SoftNetModels/01"
# for folder_name in os.listdir(obj_dir):
#     if folder_name.startswith('a'):
#         obj_path = os.path.join(obj_dir, folder_name, "model.obj")
#         vertices, faces = read_obj(obj_path)
#         face_normals = calculate_normals(vertices, faces)
#         vertex_normals = calculate_vertex_normals(vertices, faces, face_normals)
#         std_deviation, max_variation = normal_variation(face_normals, faces)
#         std_deviations.append(std_deviation)
#         max_variations.append(max_variation)

# pred
#obj_dir = "/root/autodl-tmp/SDFusion/test_results_nopre_large_dset/obj"
obj_dir = "/content/drive/My Drive/Colab Notebooks/Soft-Fusion-FYP/data/Softnet/01/fold based extension 1dof soft actuator 6 folds short Model 1/model.obj"
# for obj_name in os.listdir(obj_dir):
#     obj_path = os.path.join(obj_dir, obj_name)
vertices, faces = read_obj(obj_dir) # was obj_path
face_normals = calculate_normals(vertices, faces)
vertex_normals = calculate_vertex_normals(vertices, faces, face_normals)
std_deviation, max_variation = normal_variation(face_normals, faces)
if np.isnan(std_deviation) or np.isnan(max_variation):
    print(obj_name)
else:
    std_deviations.append(std_deviation)
    max_variations.append(max_variation)

print(f"Average Standard Deviation of Normal Variation: {np.mean(std_deviations)}")
print(f"Average Maximum Angle Variation: {np.mean(max_variations)}")