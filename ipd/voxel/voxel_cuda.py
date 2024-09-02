import os
import ipd

_voxel = ipd.cuda.build_extension("_voxel", [
    os.path.abspath(f"{os.path.dirname(__file__)}/_voxel.cpp"),
    os.path.abspath(f"{os.path.dirname(__file__)}/_voxel.cu"),
], ['voxel'], globals())

