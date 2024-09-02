import sys
import os
from pathlib import Path
from importlib import import_module
from types import ModuleType
from typing import List

class LazyModule:
    __slots__ = ('_name', '_package')

    def __init__(
        self,
        name: str,
        package: str = None,
    ):
        self._name = name
        self._package = package or name.split('.', maxsplit=1)[0]

    def _import_module(self) -> ModuleType:
        try:
            return import_module(name=self._name)
        except ImportError as e:
            print('-'*80)
            print(f'import_module({self._name}) failed, sys.path is:')
            for p in sys.path:
                p = Path(p) / self._name.replace('.', '/')
                print(p)
                if os.path.exists(p):
                    print('WTF, path exists:')
                    for f in os.listdir(p):
                        print(f'    {f}')
            print(e)
            print('-'*80)
            raise ImportError(f'Failed to import module: {self._name}')
        except Exception:
            raise ImportError(f'Failed to import module: {self._name}')

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
