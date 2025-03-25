import os
import ipd

try:
    _rms = ipd.dev.cuda.build_extension("_rms", [
        os.path.abspath(f"{os.path.dirname(__file__)}/_qcp_rms.cpp"),
        os.path.abspath(f"{os.path.dirname(__file__)}/_qcp_rms.cu"),
    ], ['fit'], globals())
except OSError as e:
    raise ImportError(f'cant import voxel_cuda, build error: {str(e)}') from e
