#!/bin/bash

. /content/drive/MyDrive/Colab\ Notebooks/Soft-Fusion-FYP/preprocess/isosurface/LIB_PATH
dset='softnet'
computeDistanceField="/content/drive/MyDrive/Colab Notebooks/Soft-Fusion-FYP/preprocess/isosurface/computeDistanceField"
# reduce=2 for 128x128x128, reduce=4 for 64x64x64
reduce=4
# category='all'
category='robot'
python -u /content/drive/MyDrive/Colab\ Notebooks/Soft-Fusion-FYP/preprocess/create_sdf.py --dset ${dset} --thread_num 128 --reduce ${reduce} --category ${category}