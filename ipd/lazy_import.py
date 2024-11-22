import subprocess
import sys
from importlib import import_module
from types import ModuleType
import typing

def lazyimport(name: str,
               package: str = '',
               pip: bool = False,
               mamba: bool = False,
               channels: str = '',
               warn: bool = True) -> ModuleType:
    """Lazy import of a module. The module will be imported when it is first accessed.

    Args:
        name (str): The name of the module to import.
        package (str): The package to install if the module cannot be imported.
        pip (bool): If True, try to install the package globally, then with --user, using pip.
        mamba (bool): If True, try to install the package globally using mamba.
        channels (str): The conda channels to use when installing the package.
        warn (bool): If True, print a warning if the module cannot be imported.

    """
    if typing.TYPE_CHECKING:
        return import_module(name)
    return _LazyModule(name, package, pip, mamba, channels, warn)

class _LazyModule:
    """A class to represent a lazily imported module."""
    __slots__ = ('_name', '_package', '_pip', '_mamba', '_channels', '_callerinfo', '_warn')

    def __init__(self, name: str, package: str = '', pip=False, mamba=False, channels='', warn=True):
        from ipd.dev.code.inspect import caller_info
        self._name = name
        self._package = package or name.split('.', maxsplit=1)[0]
        self._pip = pip
        self._mamba = mamba
        self._channels = channels
        self._callerinfo = caller_info(excludefiles=[__file__])
        self._warn = warn
        # if name not in _DEBUG_ALLOW_LAZY_IMPORT:
        #     self.now()
        #     _all_skipped_lazy_imports.add(name)

    def now(self) -> ModuleType:
        """Import the module now."""
        try:
            return self._mambathenpipimport()
        # except ImportError as e:
        # msg = f'lazy import of module {self._name} failed, continuing without {self._name} support'
        # if msg not in _warned:
        # print(msg)
        # _warned.add(msg)
        # for p in sys.path:
        # print(p)
        # raise ImportError(f'Failed to import module: {self._name}') from e
        # raise e from e
        except Exception as e:
            callinfo = f'{self._callerinfo.filename}:{self._callerinfo.lineno}\n    {self._callerinfo.code}'
            if self._warn:
                print(f'_LazyModule: Failed to import module: {self._name}\nFile: {callinfo}', flush=True)
            raise e

    def _mambathenpipimport(self):
        try:
            return import_module(self._name)
        except (ValueError, AssertionError, ModuleNotFoundError):

            if self._mamba:
                self._try_mamba_install()
            try:
                return import_module(self._name)
            except (ValueError, AssertionError, ModuleNotFoundError):
                return self._pipimport()

    def _try_mamba_install(self):
        mamba = sys.executable.replace('/bin/python', '')
        mamba, env = mamba.split('/')
        # mamba = '/'.join(mamba[:-1])+'/bin/mamba'
        mamba = 'mamba'
        cmd = f'{mamba} activate {env} && {mamba} install {self._channels} {self._package}'
        result = subprocess.check_call(cmd.split(), shell=True)
        assert not isinstance(result, int) and 'error' not in result.lower()

    def _pipimport(self):
        global _skip_global_install
        try:
            return import_module(self._name)
        except (ValueError, AssertionError, ModuleNotFoundError):
            if self._pip and self._pip != 'user':
                if not _skip_global_install:
                    try:
                        sys.stderr.write(f'PIPIMPORT {self._package}\n')
                        result = subprocess.check_call(f'{sys.executable} -mpip install {self._package}'.split())
                    except:  # noqa
                        pass
            try:
                return import_module(self._name)
            except (ValueError, AssertionError, ModuleNotFoundError):
                if self._pip and self._pip != 'nouser':
                    _skip_global_install = True
                    sys.stderr.write(f'PIPIMPORT --user {self._package}\n')
                    try:
                        result = subprocess.check_call(f'{sys.executable} -mpip install --user {self._package}'.split())
                        sys.stderr.write(str(result))
                    except:  # noqa
                        pass
                return import_module(self._name)

    @property
    def _module(self) -> ModuleType:
        return sys.modules.get(self._name) or self.now()

    def __getattr__(self, name: str):
        return getattr(self._module, name)

    def __dir__(self) -> list[str]:
        return dir(self._module)

    def __repr__(self) -> str:
        return '{t}({n})'.format(
            t=type(self).__name__,
            n=self._name,
        )

_all_skipped_lazy_imports = set()
_skip_global_install = False
_warned = set()

# from ipd.dev.contexts import onexit
# @onexit
# def print_skipped():
#     if _all_skipped_lazy_imports:
#         print(_all_skipped_lazy_imports)

_DEBUG_ALLOW_LAZY_IMPORT = [
    'ipd.crud',
    'ipd.dev.cuda',
    'ipd.observer',
    'ipd.dev.qt',
    'ipd.dev.sieve',
    'ipd.fit',
    'ipd.motif',
    'ipd.pdb',
    'ipd.samp',
    'ipd.samp.sampling_cuda',
    'ipd.protocol',
    'ipd.sym',
    'ipd.tests',
    'ipd.tools',
    'ipd.viz',
    'ipd.viz.viz_pdb',
    'ipd.voxel',
    'pymol',
    'pymol.cgo',
    'pymol.cmd',
    'sqlmodel',
    'fastapi',
    'torch',
    'ipd.sym.high_t',
    'omegaconf',
    'ipd.dev.cli',
    'hydra',
    'ipd.sym.sym_tensor',
    'ipd.homog',
    'ipd.sym.xtal',
    'RestricetedPython',
    'ipd.homog.thgeom',
    'ipd.homog.quat',
    'ipd.sym.helix',
    'ipd.dev.testing',
    'ipd.tests.sym',
]
