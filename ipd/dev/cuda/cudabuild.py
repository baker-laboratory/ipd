import pytest

th = pytest.importorskip('torch')
pytest.mark.skipif(not th.cuda.device_count(), 'cuda unavailable')

import os

import ipd
from ipd.dev.error import change_exception

mode = 'release'
# mode = 'debug'
# 16.2 vs 20 for voxdock test
# 1.62ms 1.5909 randxform(1e6) wo
# 1.47ms 1.4576 with
# 32.2ms 25.8 randxform(2**24) without DEIGEN_NO_DEBUG
# 28.4ms randxform(2**24) with DEIGEN_NO_DEBUG

os.environ['TORCH_CUDA_ARCH_LIST'] = ''

@change_exception(OSError=ImportError, RuntimeError=ImportError)  # type: ignore
def build_extension(name, sources, incpath, module=None, verbose=False):
    """Build a torch extension module from sources.

    Args:
        name (str): Name of the extension module.
        sources (list): List of source files.
        incpath (list): List of include paths.
        module (module, optional): Module to add extension to.
        verbose (bool, optional): Print verbose output.
    Returns:
        Extension module.
    """
    import torch as th  # type: ignore
    import torch.utils.cpp_extension  # type: ignore
    # os.environ['CC'] = "gcc-9"
    commonflags = ['-DEIGEN_NO_DEBUG'] if mode == 'release' else []
    # ic('start cuda build')
    I = [
        # '/home/sheffler/sw/MambaForge/envs/TEST/lib/python3.12/site-packages/numba/cuda/',
        # f'{os.path.dirname(th.__file__)}/include/torch/csrc/api/include/torch',
    ]
    extension = th.utils.cpp_extension.load(  # type: ignore
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
