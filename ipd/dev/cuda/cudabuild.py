import pytest

th = pytest.importorskip('torch')
pytest.mark.skipif(not th.cuda.device_count(), 'cuda unavailable')

import os

import ipd

mode = 'release'
# mode = 'debug'
# 16.2 vs 20 for voxdock test
# 1.62ms 1.5909 randxform(1e6) wo
# 1.47ms 1.4576 with
# 32.2ms 25.8 randxform(2**24) without DEIGEN_NO_DEBUG
# 28.4ms randxform(2**24) with DEIGEN_NO_DEBUG

os.environ['TORCH_CUDA_ARCH_LIST'] = ''

@ipd.dev.change_exception(OSError=ImportError, RuntimeError=ImportError)
def build_extension(name, sources, incpath, module=None, verbose=False):
    import torch as th
    import torch.utils.cpp_extension
    # os.environ['CC'] = "gcc-9"
    commonflags = ['-DEIGEN_NO_DEBUG'] if mode == 'release' else []
    # ic('start cuda build')
    I = [
        # '/home/sheffler/sw/MambaForge/envs/TEST/lib/python3.12/site-packages/numba/cuda/',
        # f'{os.path.dirname(th.__file__)}/include/torch/csrc/api/include/torch',
    ]
    extension = th.utils.cpp_extension.load(
        name=name,
        sources=sources,
        is_python_module=True,
        # verbose=True,
        verbose=verbose,
        extra_cflags=['-O3'] + commonflags,
        extra_cuda_cflags=['-Xnvlink', '-use-host-info'] + commonflags,
        extra_include_paths=[f'{ipd.projdir}/{d}' for d in ['../lib', 'cuda'] + incpath] + I)
    # ic('done cuda build')
    if module:
        for k, v in extension.__dict__.items():
            if not k.startswith('__'):
                module[k] = v
    return extension
