import os
import ipd

try:
    ipd.dev.cuda.build_extension("_sampling", [
        os.path.abspath(f"{os.path.dirname(__file__)}/_sampling.cpp"),
        os.path.abspath(f"{os.path.dirname(__file__)}/_sampling.cu"),
    ], ['samp'], globals())
except OSError as e:
    raise ImportError(f'cant import voxel_cuda, build error: {str(e)}') from e
