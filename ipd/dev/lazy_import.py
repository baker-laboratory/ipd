import sys
import os
from pathlib import Path
from importlib import import_module
from types import ModuleType
from typing import List
import subprocess

class LazyModule:
    __slots__ = ('_name', '_package', '_pip', '_mamba', '_channels')

    def __init__(self, name: str, package: str = None, pip: str | bool = False, mamba=False, channels=''):
        self._name = name
        self._package = package or name.split('.', maxsplit=1)[0]
        self._pip = pip
        self._mamba = mamba
        self._channels = channels

    def _import_module(self) -> ModuleType:
        try:
            return self._mambathenpipimport()
        except ImportError as e:
            # print('-' * 80)
            print(f'import_module({self._name}) failed')
            # print({self._package} {self._pip} sys.path is:')
            for p in sys.path:
                p = Path(p) / self._name.replace('.', '/')
                # print(p)
                if os.path.exists(p):
                    print('WTF, path exists:')
                    for f in os.listdir(p):
                        print(f'    {f}')
            # print(e)
            # print('-' * 80)
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
                subprocess.check_call(cmd.split(), shell=True)
            try:
                return import_module(self._name)
            except (ValueError, AssertionError, ModuleNotFoundError):
                return self._pipimport()

    def _pipimport(self):
        try:
            return import_module(self._name)
        except (ValueError, AssertionError, ModuleNotFoundError):
            if self._pip and self._pip != 'user':
                subprocess.check_call(f'{sys.executable} -mpip install {self._package}'.split())
            try:
                return import_module(self._name)
            except (ValueError, AssertionError, ModuleNotFoundError):
                if self._pip and self._pip != 'nouser':
                    subprocess.check_call(f'{sys.executable} -mpip install --user {self._package}'.split())
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
