import inspect
import sys
from importlib import import_module
from types import ModuleType
import typing
from .import_util import is_installed

forbid = set()

def lazyimports(
        *names: str,
        package: typing.Sequence[str] = (),
        **kw,
) -> list[ModuleType]:
    """Lazy import of a module. The module will be imported when it is first accessed.

    Args:
        names (str): The name(s) of the module(s) to import.
        package (str): The package to install if the module cannot be imported.
        warn (bool): If True, print a warning if the module cannot be imported.

    """
    assert len(names)
    if not names: raise ValueError('package name is required')
    if package: assert len(package) == len(names) and not isinstance(package, str)
    else: package = ('', ) * len(names)
    modules = [lazyimport(name, package=pkg, **kw) for name, pkg in zip(names, package)]
    return modules

def timed_import_module(modnames):
    if isinstance(modnames, str): modnames = (modnames, )
    for modname in modnames:
        assert modname not in forbid, f'forbbiden import! {modname}'
        try:
            mod = import_module(modname)
            break
        except ImportError:
            mod = None
    if mod is None: import_module(modnames[0])
    return mod

def lazyimport(name: 'str | tuple[str]',
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
            return FalseModule(name if isinstance(name, str) else name[0])
    else:
        return _LazyModule(name, package, pip, mamba, channels, warn)

def maybeimport(name) -> ModuleType:
    return lazyimport(name, maybeimport=True)

def maybeimports(*names) -> list[ModuleType]:
    return lazyimports(*names, maybeimport=True)

class LazyImportError(ImportError):
    pass

def _get_package(name):
    if isinstance(name, str): return name.split('.', maxsplit=1)[0]
    if isinstance(name, tuple): return tuple(n.split('.', maxsplit=1)[0] for n in name)
    raise ValueError(f'Invalid name type: {type(name)}')

class _LazyModule(ModuleType):
    """A class to represent a lazily imported module."""

    # __slots__ = ('_lazymodule_name', '_lazymodule_package', '_lazymodule_pip', '_lazymodule_mamba', '_lazymodule_channels', '_lazymodule_callerinfo', '_lazymodule_warn')

    def __init__(self, name: str, package: str = '', pip=False, mamba=False, channels='', warn=True):
        # from ipd.dev.code.inspect import caller_info
        self._lazymodule_name = name
        self._lazymodule_package = package or _get_package(name)
        self._lazymodule_pip = pip
        self._lazymodule_mamba = mamba
        self._lazymodule_channels = channels
        # self._lazymodule_callerinfo = caller_info(excludefiles=[__file__])
        self._lazymodule_warn = warn
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
                    return FalseModule(self._lazymodule_name if isinstance(self._lazymodule_name, str) else self._lazymodule_name[0])
            raise e from None


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

    # def _try_mamba_install(self):
    #     mamba = sys.executable.replace('/bin/python', '')
    #     mamba, env = mamba.split('/')
    #     # mamba = '/'.join(mamba[:-1])+'/bin/mamba'
    #     mamba = 'mamba'
    #     cmd = f'{mamba} activate {env} && {mamba} install {self._lazymodule_channels} {self._lazymodule_package}'
    #     result = subprocess.check_call(cmd.split(), shell=True)
    #     assert not isinstance(result, int) and 'error' not in result.lower()

    # def _pipimport(self):
    #     global _skip_global_install
    #     try:
    #         return timed_import_module(self._lazymodule_name)
    #     except (ValueError, AssertionError, ModuleNotFoundError):
    #         if self._lazymodule_pip and self._lazymodule_pip != 'user':
    #             if not _skip_global_install:
    #                 try:
    #                     sys.stderr.write(f'PIPIMPORT {self._lazymodule_package}\n')
    #                     result = subprocess.check_call(
    #                         f'{sys.executable} -mpip install {self._lazymodule_package}'.split())
    #                 except:  # noqa
    #                     pass
    #         try:
    #             return timed_import_module(self._lazymodule_name)
    #         except (ValueError, AssertionError, ModuleNotFoundError):
    #             if self._lazymodule_pip and self._lazymodule_pip != 'nouser':
    #                 _skip_global_install = True
    #                 sys.stderr.write(f'PIPIMPORT --user {self._lazymodule_package}\n')
    #                 try:
    #                     result = subprocess.check_call(
    #                         f'{sys.executable} -mpip install --user {self._lazymodule_package}'.split())
    #                     sys.stderr.write(str(result))
    #                 except:  # noqa
    #                     pass
    #             return timed_import_module(self._lazymodule_name)
