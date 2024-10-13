import sys
import os
from pathlib import Path
from importlib import import_module
from types import ModuleType
from typing import List
import subprocess

_skip_global_install = False

_warned = set()

class LazyModule:
    __slots__ = ('_name', '_package', '_pip', '_mamba', '_channels')

    def __init__(self, name: str, package: str = None, pip=False, mamba=False, channels=''):
        self._name = name
        self._package = package or name.split('.', maxsplit=1)[0]
        self._pip = pip
        self._mamba = mamba
        self._channels = channels

    def _import_module(self) -> ModuleType:
        try:
            return self._mambathenpipimport()
        except ImportError as e:
            msg = f'lazy import of module {self._name} failed, continuing without {self._name} support'
            if msg not in _warned:
                print(msg)
                _warned.add(msg)
            # for p in sys.path:
            # print(p)
            raise ImportError(f'Failed to import module: {self._name}') from e
        except Exception as e:
            raise ImportError(f'Failed to import module: {self._name}') from e

    def _mambathenpipimport(self):
        try:
            return import_module(self._name)
        except (ValueError, AssertionError, ModuleNotFoundError):

            if self._mamba:
                mamba = sys.executable.replace('/bin/python', '')
                *mamba, env = mamba.split('/')
                # mamba = '/'.join(mamba[:-1])+'/bin/mamba'
                mamba = 'mamba'
                cmd = f'{mamba} activate {env} && {mamba} install {self._channels} {self._package}'
                result = subprocess.check_call(cmd.split(), shell=True)
                assert 'error' not in result.lower()
            try:
                return import_module(self._name)
            except (ValueError, AssertionError, ModuleNotFoundError):
                return self._pipimport()

    def _pipimport(self):
        global _skip_global_install
        try:
            return import_module(self._name)
        except (ValueError, AssertionError, ModuleNotFoundError):
            if self._pip and self._pip != 'user':
                if not _skip_global_install:
                    try:
                        sys.stderr.write(f'PIPIMPORT {self._package}\n')
                        result = subprocess.check_call(
                            f'{sys.executable} -mpip install {self._package}'.split())
                    except:
                        pass
            try:
                return import_module(self._name)
            except (ValueError, AssertionError, ModuleNotFoundError):
                if self._pip and self._pip != 'nouser':
                    _skip_global_install = True
                    sys.stderr.write(f'PIPIMPORT --user {self._package}\n')
                    try:
                        result = subprocess.check_call(
                            f'{sys.executable} -mpip install --user {self._package}'.split())
                        sys.stderr.write(result)
                    except:
                        pass
                return import_module(self._name)

    @property
    def _module(self) -> ModuleType:
        return sys.modules.get(self._name) or self._import_module()

    def __getattr__(self, name: str):
        return getattr(self._module, name)

    def __dir__(self) -> List[str]:
        return dir(self._module)

    def __repr__(self) -> str:
        return '{t}({n})'.format(
            t=type(self).__name__,
            n=self._name,
        )

def lazyimport(*a, **kw):
    return LazyModule(*a, **kw)
