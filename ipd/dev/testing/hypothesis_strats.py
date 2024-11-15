import datetime
import types
import uuid
from pathlib import Path
from typing import Any, Optional, Type, Union, _AnnotatedAlias, get_args  # type: ignore
import pydantic
import pydantic_core
from hypothesis import strategies as st

class PydanticStrats:
    def __init__(self, overrides=None, type_mapping=None, exclude_attrs=None):
        self.overrides = overrides or {}
        # self.urls_strat = st.text().filter(str.isidentifier).map(lambda s: f'https://example.com/{s}.git').map(
        # pydantic.AnyUrl)
        self.files_strat = st.sampled_from(['pyproject.toml', '.gitignore']).map(Path)
        self.urlstrat = st.just('https://github.com/baker-laboratory/ipd.git')
        self.type_mapping = {
            str: st.text(),
            int: st.integers(),
            float: st.floats(),
            bool: st.booleans(),
            list[int]: st.lists(st.integers()),
            list[str]: st.lists(st.text()),
            dict[str, int]: st.dictionaries(st.text(), st.integers()),
            dict[str, list[str]]: st.dictionaries(st.text(), st.lists(st.text())),
            uuid.UUID: st.uuids(),
            datetime.datetime: st.datetimes(),
            pydantic_core._pydantic_core.Url: self.urlstrat,
            Path: self.files_strat,
            # pydantic_core._pydantic_core.Url: provisional.urls,
            # pydantic_core._pydantic_core.Url: urls_strat,
            # Path: st.text().map(Path),
        }
        self.exclude_attrs = exclude_attrs or set()
        self.type_mapping.update(type_mapping or {})

    def __call__(self, Model: Type[pydantic.BaseModel]) -> st.SearchStrategy[dict[str, Any]]:
        strategy_dict = {'__test__': st.just(True)}
        for attr, field in Model.model_fields.items():
            field_type = field.annotation
            if (strategy := self.get_strategy(Model, attr, field_type)) is not None:
                strategy = self.postprocess_field_strategy(Model, attr, field, strategy)
                assert isinstance(strategy, st.SearchStrategy)
                if field.default is not pydantic_core.PydanticUndefined:
                    strategy = st.one_of(st.just(field.default), strategy)
                strategy_dict[attr] = strategy
        modelstrat = st.fixed_dictionaries(strategy_dict)
        modelstrat = self.postprocess_moodel_strategy(Model, modelstrat)
        assert isinstance(modelstrat, st.SearchStrategy)
        return modelstrat

    def pick_from_union_types(self, Ts):
        Ts = [T for T in Ts if T is not type(None)]
        if len(Ts) == 1: return Ts[0]
        if Ts == [uuid.UUID, str]: return uuid.UUID

    def get_strategy(self, Model, attr, T):
        if attr in self.exclude_attrs: return None
        if None is not (strat := self.overrides.get(f'*.{attr}')): return strat
        if None is not (strat := self.overrides.get(f'{Model.__name__}.{attr}')): return strat
        orig, args = getattr(T, '__origin__', None), get_args(T)
        if orig is Optional: T = args[0]
        if orig is Union: T = self.pick_from_union_types(args)
        if orig is list and issubclass(args[0], pydantic.BaseModel): return None
        if isinstance(T, types.UnionType): T = self.pick_from_union_types(T.__args__)
        if isinstance(T, _AnnotatedAlias): T = get_args(args[0])[0]
        if (strat := self.type_mapping.get(T, None)) is not None: return strat  # type: ignore
        raise Valuerror(f'cant make hypothesis strat for {T} {type(T)} {orig} {args}')  # type: ignore

    def postprocess_moodel_strategy(self, Model, strat):
        return strat

    def postprocess_field_strategy(self, Model, attr, field, strat):
        return strat
