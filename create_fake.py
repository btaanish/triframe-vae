obj_path = '/root/autodl-tmp/SDFusion/test_results_pretrained_small_dset/obj'
out_path = '/root/autodl-tmp/SDFusion/data/ShapeNet/ShapeNetCore.v1/01'

import os
for i, obj in enumerate(os.listdir(obj_path)):
    outname = f'b{i:03d}'
    os.makedirs(os.path.join(out_path, outname), exist_ok=True)
    os.system(f'cp "{os.path.join(obj_path, obj)}" {os.path.join(out_path, outname)}')
    os.system(f'mv "{os.path.join(out_path, outname)}/{obj}" {os.path.join(out_path, outname)}/model.obj')
    print(outname)