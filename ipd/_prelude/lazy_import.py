forbid = set()  # for debugging stuff that is being incorrectly imported

import inspect
import sys
from importlib import import_module
from types import ModuleType
import typing
from .import_util import is_installed

def lazyimport(name: 'str | tuple[str]',
               package: str = '',
               warn: bool = True,
               maybeimport=False) -> ModuleType:
    return _LazyModule(name, package, warn=warn, maybe=maybeimport)

def lazyimports(
        *names: str,
        package: typing.Sequence[str] = (),
        **kw,
) -> list[ModuleType]:
    """Lazy import of a module. The module will be imported when it is first accessed.

    Args:
        names (str): The name(s) of the module(s) to import.
        package (str): The package to install if the module cannot be imported.
    """
    assert len(names)
    if len(names) == 0: raise ValueError('package name is required')
    if package: assert len(package) == len(names) and not isinstance(package, str)
    else: package = ('', ) * len(names)
    modules = [lazyimport(name, package=pkg, **kw) for name, pkg in zip(names, package)]
    return modules

def maybeimport(name) -> ModuleType:
    try:
        return import_module(name)
    except ImportError:
        return FalseModule(name)

def maybeimports(*names) -> list[ModuleType]:
    if len(names) == 1 and isinstance(names, str):
        names = names.split()
    return [maybeimport(name) for name in names]
def timed_import_module(modnames):
    import evn
    if isinstance(modnames, str): modnames = (modnames, )
    mod = None
    for modname in modnames:
        assert modname not in forbid, f'forbbiden import! {modname}'
        try:
            evn.chrono_enter_scope(f'lazyimport {modname}')
            mod = import_module(modname)
            break
        except ImportError:
            mod = None
        finally:
            evn.chrono_exit_scope(f'lazyimport {modname}')
    if mod is None: import_module(modnames[0])  # to raise ImportError
    return mod


class LazyImportError(ImportError):
    pass

def _get_package(name):
    if isinstance(name, str): return name.split('.', maxsplit=1)[0]
    if isinstance(name, tuple): return tuple(n.split('.', maxsplit=1)[0] for n in name)
    raise ValueError(f'Invalid name type: {type(name)}')

class _LazyModule(ModuleType):
    """A class to represent a lazily imported module."""

    # __slots__ = ('_lazymodule_name', '_lazymodule_package', '_lazymodule_pip', '_lazymodule_mamba', '_lazymodule_channels', '_lazymodule_callerinfo', '_lazymodule_warn')

    def __init__(self, name: str | tuple[str], package: str = '', warn=True, maybe=False):
        # from ipd.dev.code.inspect import caller_info
        self._lazymodule_name = name
        self._lazymodule_package = package or _get_package(name)
        # self._lazymodule_pip = pip
        # self._lazymodule_mamba = mamba
        # self._lazymodule_channels = channels
        # self._lazymodule_callerinfo = caller_info(excludefiles=[__file__])
        self._lazymodule_warn = warn
        self._lazymodule_maybe = maybe
        # if name not in _DEBUG_ALLOW_LAZY_IMPORT:
        #     self._lazymodule_now()
        #     _all_skipped_lazy_imports.add(name)

    def _lazymodule_import_now(self) -> ModuleType:
        """Import the module _lazymodule_import_now."""
        try:
            return timed_import_module(self._lazymodule_name)
        except ImportError as e:
            if 'doctest' in sys.modules:
                if in_doctest():
                    return FalseModule(self._lazymodule_name if isinstance(self._lazymodule_name, str
                                                                           ) else self._lazymodule_name[0])
            # ci = self._lazymodule_callerinfo
            # callinfo = f'\n  File "{ci.filename}", line {ci.lineno}\n    {ci.code}'
            raise e

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

def in_doctest():
    return any('doctest' in frame.filename for frame in inspect.stack())

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
#     'ipd.cuda',
#     'ipd.observer',
#     'ipd.dev.qt',
#     'ipd.dev.sieve',
#     'ipd.cuda.rms',
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
#     'ipd.cuda.voxel',
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
