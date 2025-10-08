# convert .obj files to .hdf5 files
import os
import h5py
import numpy as np

def convert_obj_to_hdf5(obj_path, hdf5_path):
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(list(map(float, line.strip().split()[1:])))
        # elif line.startswith('f '):
        #     faces.append(list(map(int, line.strip().split()[1:])))
    vertices = np.array(vertices)

    faces = np.array(faces)

    if vertices.shape[-1] == 3:
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('vertices', data=vertices)
            f.create_dataset('faces', data=faces)
    else:
        raise ValueError(f'Vertices must have 3 coordinates: bad file {obj_path}')


if __name__ == '__main__':
    # obj_dir = 'test_results_pretrained_small_dset/obj'
    # hdf5_dir = 'test_results_pretrained_small_dset/hdf5'


    # for obj_name in os.listdir(obj_dir):
    #     obj_path = os.path.join(obj_dir, obj_name)
    #     hdf5_path = os.path.join(hdf5_dir, obj_name.replace('.obj', '.hdf5'))
    #     convert_obj_to_hdf5(obj_path, hdf5_path)
    #     print('Converted', obj_name)

    # obj_dir = "/root/autodl-tmp/SDFusion/data/ShapeNet/SoftNetModels/01"
    obj_dir = "/root/autodl-tmp/SDFusion/test_results_nopre_small_dset/obj"
    hdf5_dir = "/root/autodl-tmp/SDFusion/test_results_nopre_small_dset/hdf5"

    # for folder_name in os.listdir(obj_dir):
    #     if folder_name.startswith('a'):
    #         obj_path = os.path.join(obj_dir, folder_name, "model.obj")
    #         hdf5_path = os.path.join(hdf5_dir, folder_name + ".hdf5")
    #         print(obj_path)
    #         convert_obj_to_hdf5(obj_path, hdf5_path)
    #         print('Converted', folder_name)

    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)

    for obj_name in os.listdir(obj_dir):
        obj_path = os.path.join(obj_dir, obj_name)
        hdf5_path = os.path.join(hdf5_dir, obj_name.replace('.obj', '.hdf5'))
        convert_obj_to_hdf5(obj_path, hdf5_path)
        print('Converted', obj_name)