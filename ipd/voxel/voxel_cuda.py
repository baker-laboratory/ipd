import os
import ipd

try:
    _voxel = ipd.dev.cuda.build_extension("_voxel", [
        os.path.abspath(f"{os.path.dirname(__file__)}/_voxel.cpp"),
        os.path.abspath(f"{os.path.dirname(__file__)}/_voxel.cu"),
    ], ['voxel'], globals())
except OSError as e:
    raise ImportError(f'cant import voxel_cuda, build error: {str(e)}') from e
