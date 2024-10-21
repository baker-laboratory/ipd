import ipd
import typer
from ipd.dev.types import KW
from typing import Self
# from ipd.dev.tools.pkginfo import cwd_package, PkgInfo

CB = type['CliBase']

typer_args = dict(no_args_is_help=True, pretty_exceptions_enable=False)

class CliBase:
    @classmethod
    def mrca(_, classes: set[CB], cls: CB | None = None) -> CB | None:
        classes = set(classes)
        cls = cls or _.__root__
        for c in cls.__children__:
            if val := cls.mcra(classes, c): return val
        if classes.issubset(cls.__descendants__):
            return cls

    @classmethod
    def __add_all_cmds__(cls: CB, self: Self, **kw: KW) -> None:
        cmds = [m for m in cls.__dict__ if callable(getattr(cls, m)) and m[0] != '_']
        for attr in cmds:
            with ipd.dev.cast(cls, self) as newself:
                # print('add command', cls, attr)
                method = getattr(newself, attr)
                setattr(cls, attr, cls.__app__.command(attr)(method))
                # setattr(CliBase, attr, getattr(cls, attr))

    @classmethod
    def __set_relationships__(cls: CB, **kw: KW) -> None:
        parent: list[CB] = [b for b in cls.__bases__ if hasattr(b, '__app__')]
        assert len(parent) < 2
        cls.__app__: typer.Typer = typer.Typer(**typer_args)
        cls.__parent__: CB | None = parent[0] if parent else None
        cls.__root__ = cls.__parent__.__root__ if parent else cls
        cls.__children__: set[CB] = set()
        cls.__siblings__: set[CB] = set()
        cls.__ancestors__: set[CB] = {cls.__parent__} if parent else set()
        cls.__descendants__: set[CB] = set()
        cls.__nesting_cmds__: list[str] = []
        if parent:
            cls.__siblings__ |= cls.__parent__.__children__
            for sibling in cls.__siblings__:
                sibling.__siblings__.add(cls)
            cls.__parent__.__children__.add(cls)
            cls.__ancestors__ |= cls.__parent__.__ancestors__
            for ancestor in cls.__ancestors__:
                ancestor.__descendants__.add(cls)
            cls.__parent__.__app__.add_typer(cls.__app__, name=cls.__name__.replace('Tool', '').lower())

    def __init_subclass__(cls: CB, **kw: KW):
        assert not cls.__module__.startswith('__main__')
        super().__init_subclass__(**kw)
        cls.__set_relationships__(**kw)

    def __init__(self, **kw):
        self.__class__.__add_all_cmds__(self, **kw)
        for cls in self.__descendants__:
            cls.__add_all_cmds__(self, **kw)
            cls.__init__(self, **kw)

    def run(self):
        self.__root__.__app__()
