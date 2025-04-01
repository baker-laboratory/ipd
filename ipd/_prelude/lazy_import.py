import subprocess
import sys
from importlib import import_module
from types import ModuleType
import typing
from .import_util import is_installed

def lazyimports(
        *names: str,
        package: typing.Sequence[str] = (),
        **kw,
) -> list[ModuleType]:
    """Lazy import of a module. The module will be imported when it is first accessed.

    Args:
        names (str): The name(s) of the module(s) to import.
        package (str): The package to install if the module cannot be imported.
        pip (bool): If True, try to install the package globally, then with --user, using pip.
        mamba (bool): If True, try to install the package globally using mamba.
        channels (str): The conda channels to use when installing the package.
        warn (bool): If True, print a warning if the module cannot be imported.

    """
    assert len(names)
    if len(names) == 0: raise ValueError('package name is required')
    if package: assert len(package) == len(names) and not isinstance(package, str)
    else: package = ('', ) * len(names)
    modules = [lazyimport(name, package=pkg, **kw) for name, pkg in zip(names, package)]
    return modules

def timed_import_module(name):
    import ipd
    ipd.dev.global_timer.checkpoint(interject=True)
    mod = import_module(name)
    ipd.dev.global_timer.checkpoint(f'LAZY import {name}')
    return mod

def lazyimport(name: str,
               package: str = '',
               pip: bool = False,
               mamba: bool = False,
               channels: str = '',
               warn: bool = True,
               maybeimport=False) -> ModuleType:
    if typing.TYPE_CHECKING or maybeimport:
        try:
            return timed_import_module(name)
        except ImportError:
            return FalseModule(name)
    else:
        return _LazyModule(name, package, pip, mamba, channels, warn)

def maybeimport(name) -> ModuleType:
    return lazyimport(name, maybeimport=True)

def maybeimports(*names) -> list[ModuleType]:
    return lazyimports(*names, maybeimport=True)

class LazyImportError(ImportError):
    pass

class _LazyModule(ModuleType):
    """A class to represent a lazily imported module."""

    # __slots__ = ('_lazymodule_name', '_lazymodule_package', '_lazymodule_pip', '_lazymodule_mamba', '_lazymodule_channels', '_lazymodule_callerinfo', '_lazymodule_warn')

    def __init__(self, name: str, package: str = '', pip=False, mamba=False, channels='', warn=True):
        from ipd.dev.code.inspect import caller_info
        self._lazymodule_name = name
        self._lazymodule_package = package or name.split('.', maxsplit=1)[0]
        self._lazymodule_pip = pip
        self._lazymodule_mamba = mamba
        self._lazymodule_channels = channels
        self._lazymodule_callerinfo = caller_info(excludefiles=[__file__])
        self._lazymodule_warn = warn
        # if name not in _DEBUG_ALLOW_LAZY_IMPORT:
        #     self._lazymodule_now()
        #     _all_skipped_lazy_imports.add(name)

    def _lazymodule_import_now(self) -> ModuleType:
        """Import the module _lazymodule_import_now."""
        try:
            return timed_import_module(self._lazymodule_name)
        except ImportError as e:
            if 'doctest' in sys.modules: return FalseModule(self._lazymodule_name)
            ci = self._lazymodule_callerinfo
            callinfo = f'\n  File "{ci.filename}", line {ci.lineno}\n    {ci.code}'
            raise e

    def _try_mamba_install(self):
        mamba = sys.executable.replace('/bin/python', '')
        mamba, env = mamba.split('/')
        # mamba = '/'.join(mamba[:-1])+'/bin/mamba'
        mamba = 'mamba'
        cmd = f'{mamba} activate {env} && {mamba} install {self._lazymodule_channels} {self._lazymodule_package}'
        result = subprocess.check_call(cmd.split(), shell=True)
        assert not isinstance(result, int) and 'error' not in result.lower()

    def _pipimport(self):
        global _skip_global_install
        try:
            return timed_import_module(self._lazymodule_name)
        except (ValueError, AssertionError, ModuleNotFoundError):
            if self._lazymodule_pip and self._lazymodule_pip != 'user':
                if not _skip_global_install:
                    try:
                        sys.stderr.write(f'PIPIMPORT {self._lazymodule_package}\n')
                        result = subprocess.check_call(
                            f'{sys.executable} -mpip install {self._lazymodule_package}'.split())
                    except:  # noqa
                        pass
            try:
                return timed_import_module(self._lazymodule_name)
            except (ValueError, AssertionError, ModuleNotFoundError):
                if self._lazymodule_pip and self._lazymodule_pip != 'nouser':
                    _skip_global_install = True
                    sys.stderr.write(f'PIPIMPORT --user {self._lazymodule_package}\n')
                    try:
                        result = subprocess.check_call(
                            f'{sys.executable} -mpip install --user {self._lazymodule_package}'.split())
                        sys.stderr.write(str(result))
                    except:  # noqa
                        pass
                return timed_import_module(self._lazymodule_name)

    def _lazymodule_is_loaded(self):
        return self._lazymodule_name in sys.modules

    def __getattr__(self, name: str):
        if name.startswith('_lazymodule_'): return self.__dict__[name]
        if name == '_loaded_module':
            if '_loaded_module' not in self.__dict__:
                self._loaded_module = self._lazymodule_import_now()
            return self.__dict__['_loaded_module']

        return getattr(self._loaded_module, name)

    def __dir__(self) -> list[str]:
        return dir(self._loaded_module)

    def __repr__(self) -> str:
        return '{t}({n})'.format(
            t=type(self).__name__,
            n=self._lazymodule_name,
        )

    def __bool__(self) -> bool:
        return bool(is_installed(self._lazymodule_name))

class FalseModule(ModuleType):

    def __bool__(self):
        return False

_all_skipped_lazy_imports = set()
_skip_global_install = False
_warned = set()

# from ipd.dev.contexts import onexit
# @onexit
# def print_skipped():
#     if _all_skipped_lazy_imports:
#         print(_all_skipped_lazy_imports)

# _DEBUG_ALLOW_LAZY_IMPORT = [
#     'ipd.crud',
#     'ipd.dev.cuda',
#     'ipd.observer',
#     'ipd.dev.qt',
#     'ipd.dev.sieve',
#     'ipd.fit',
#     'ipd.motif',
#     'ipd.pdb',
#     'ipd.samp',
#     'ipd.samp.sampling_cuda',
#     'ipd.protocol',
#     'ipd.sym',
#     'ipd.tests',
#     'ipd.tools',
#     'ipd.viz',
#     'ipd.viz.viz_pdb',
#     'ipd.voxel',
#     'pymol',
#     'pymol.cgo',
#     'pymol.cmd',
#     'sqlmodel',
#     'fastapi',
#     'torch',
#     'ipd.sym.high_t',
#     'omegaconf',
#     'ipd.dev.cli',
#     'hydra',
#     'ipd.sym.sym_tensor',
#     'ipd.homog',
#     'ipd.sym.xtal',
#     'RestricetedPython',
#     'ipd.homog.thgeom',
#     'ipd.homog.quat',
#     'ipd.sym.helix',
#     'ipd.dev.testing',
#     'ipd.tests.sym',
# ]
