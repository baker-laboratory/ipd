import pydantic
import sqlmodel
import sqlalchemy
import uuid
import yaml
from typing import Union, Optional, get_args
import fastapi
import rich
import inspect
from datetime import datetime
from typing import Union, Optional, Callable, Annotated, get_type_hints, Type, Any
import contextlib
import os
import uuid
import rich
import tempfile
from pathlib import Path
import ipd
from icecream import ic

_ModelRefType = Optional[Union[uuid.UUID, str]]

def _label_field(cls):
    if hasattr(cls, '_label'): return cls._label.default
    return 'name'

def validate_ref(val: Union[uuid.UUID, str], valinfo, spec_namespace):
    assert not isinstance(val, int), 'int id is wrong, use uuid now'
    if hasattr(val, 'id'): return val.id
    with contextlib.suppress(TypeError, ValueError, AttributeError):
        return uuid.UUID(val)
    specname = valinfo.config['title']
    if not specname.endswith('Spec'): specname += 'Spec'
    cls = spec_namespace[specname]
    field = cls.model_fields[valinfo.field_name]
    typehint = field.annotation
    assert typehint == _ModelRefType
    refcls = field.metadata[1]
    if isinstance(refcls, str):
        if not refcls.endswith('Spec'): refcls += 'Spec'
        refcls = spec_namespace[refcls]
    if isinstance(val, str):
        client = ipd.ppp.get_hack_fixme_global_client()
        assert client, 'client unavailable'
        refclsname = refcls.__name__.replace('Spec', '').lower()
        if not (ref := getattr(client, refclsname)(**{_label_field(refcls): val}, _ghost=True)):
            raise ValueError(f'unknown {refcls.__name__[:-4]} named "{val}"')
        val = ref.id
    # print(cls, refcls, val)
    return val

class ModelReference(type):
    def __class_getitem__(cls, T):
        outerns = inspect.currentframe().f_back.f_globals
        validator = pydantic.BeforeValidator(lambda x, y, outerns=outerns: validate_ref(x, y, outerns))
        return Annotated[Annotated[_ModelRefType, validator], T]

class SpecBase(pydantic.BaseModel):
    id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    ispublic: bool = True
    telemetry: bool = False
    ghost: bool = False
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    props: Union[list[str], str] = []
    attrs: Union[dict[str, Union[str, int, float]], str] = {}
    _errors: str = ''

    def __hash__(self):
        return self.id

    @classmethod
    def clsspec(cls):
        return cls

    def to_spec(self):
        return self

    @classmethod
    def kind(cls):
        return cls.clsspec().__name__.replace('Spec', '').lower()

    @pydantic.field_validator('props')
    def valprops(cls, props):
        if isinstance(props, (set, list)): return props
        try:
            props = ipd.dev.safe_eval(props)
        except (NameError, SyntaxError):
            if isinstance(props, str):
                if not props.strip(): return []
                props = [p.strip() for p in props.strip().split(',')]
        return props

    @pydantic.field_validator('attrs')
    def valattrs(cls, attrs):
        if isinstance(attrs, dict): return attrs
        try:
            attrs = ipd.dev.safe_eval(attrs)
        except (NameError, SyntaxError):
            if isinstance(attrs, str):
                if not attrs.strip(): return {}
                attrs = {
                    x.split('=').split(':')[0].strip(): x.split('=').split(':')[1].strip()
                    for x in attrs.strip().split(',')
                }
        return attrs

    def _copy_with_newid(self):
        return self.__class__(**{**self.model_dump(), 'id': uuid.uuid4()})

    def errors(self):
        return self._errors

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        return setattr(self, k, v)

    def print_full(self, seenit=None, depth=0):
        seenit = seenit or set()
        if self.id in seenit: return
        seenit.add(self.id)
        fields = list(self.model_fields)
        depth += 1
        if hasattr(self, '__backend_props__'): fields += self.__backend_props__
        print(self.__class__.__name__)
        for field in sorted(fields):
            prop = getattr(self, field)
            if not prop: continue
            print(' ' * depth * 4, field, '=', end=' ')
            if not isinstance(prop, (tuple, list)):
                if hasattr(prop, 'print_full'):
                    prop.print_full(seenit, depth)
                else:
                    print(prop)
            else:
                print('multiple:')
                # for p in prop:
                # if p.id not in seenit: p.print_full(seenit, depth)

class StrictFields:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        init_cls = cls.__init__.__qualname__.split('.')[-2]
        # if cls has explicit __init__, don't everride
        if init_cls == cls.__name__: return

        def __strict_init__(self, client=None, **data):
            for name in data:
                if name not in cls.__fields__:
                    raise TypeError(f"{cls} Invalid field name: {name}")
            data |= dict(client=client)
            super(cls, self).__init__(**data)

        cls.__init__ = __strict_init__

def make_client_models(spec_models, backend_models):
    print('make_client_models')
    client_models = {}
    for kind, clsspec in spec_models.items():
        clsdb = backend_models[kind]
        assert clsspec.__name__.endswith('Spec')
        clsname = clsspec.__name__[:-4]
        dbprops = set(clsdb.__dict__) | set(clsdb.model_fields)
        specprops = set(clsspec.__dict__) | set(clsspec.model_fields)
        propnames = {s for s in dbprops - specprops if not s.startswith('_')}
        props = {}
        for propname in propnames:
            if propname in clsdb.model_fields:
                proptype = clsdb.model_fields[propname].annotation
                print(clsspec, propname)
                assert 0
            elif propname in clsdb.__annotations__:
                proptype = get_args(clsdb.__annotations__[propname])[0]
            else:
                continue
            if hasattr(proptype, '__origin__'):
                propkind = get_args(proptype)[0]
            elif hasattr(proptype, '__forward_arg__'):
                propkind = proptype.__forward_arg__
            else:
                propkind = proptype.__name__
            if hasattr(propkind, '__forward_arg__'):
                propkind = propkind.__forward_arg__
            if not isinstance(propkind, str): propkind = propkind.__name__
            props[propname] = propkind.replace('DB', '').lower()
        client_models[kind] = type(clsname, (ClientModelBase, clsspec), {}, backend_props=props)
    return client_models

class ClientModelBase(pydantic.BaseModel):
    id: uuid.UUID
    _client: 'ModelFrontend' = None
    __sibling_models__: dict[str, 'ClientModelBase'] = {}

    def __init_subclass__(cls, backend_props=(), siblings=(), **kw):
        super().__init_subclass__(**kw)
        if not backend_props: return
        cls.__backend_props__ = backend_props
        cls.__sibling_models__[cls.kind()] = cls
        for attr, kind in cls.__backend_props__.items():

            def form_closure(_cls=cls, _attr=attr, _kind=kind):
                @property
                def getter(self):
                    val = self._client.getattr(_cls.kind(), self.id, _attr)
                    if val is None: return val
                    # raise AttributeError(f'kind {_cls.kind()} id {self.id} attr {_attr} is None')
                    if _kind in self.__sibling_models__:
                        attrcls = self.__sibling_models__[_kind]
                    else:
                        raise ValueError(f'unknown type {_kind}')
                    if isinstance(val, list):
                        return tuple(attrcls(self._client, **kw) for kw in val)
                    return attrcls(self._client, **val)

                return getter

            getter = form_closure()
            setattr(cls, attr, getter)

    def __init__(self, client, **kw):
        super().__init__(**kw)
        self._client = client

    def __hash__(self):
        return self.id

    def _validated(self):
        'noop, as validation should have happened at Spec stage'
        return self

    def __setattr__(self, name, val):
        assert name != 'id', 'cant set id via client'
        if self._client and name[0] != '_':
            result = self._client.setattr(self, name, val)
            assert not result, result
        super().__setattr__(name, val)

    @classmethod
    def clsspec(cls):
        for base in cls.__bases__:
            if base.__name__.endswith('Spec'): return base

    def to_spec(self):
        return self.__class__.clsspec()(**self.model_dump())

    @classmethod
    def from_spec(cls, spec):
        return cls(**spec.model_dump())

def client_obj_representer(dumper, obj):
    data = obj.dict()
    data['class'] = obj.__class__.__name__
    return dumper.represent_scalar('!Pydantic', data)

def client_obj_constructor(loader, node):
    value = loader.construct_scalar(node)
    cls = globals()[value.pop('class')]
    return cls(**value)

yaml.add_representer(ClientModelBase, client_obj_representer)
yaml.add_constructor('!Pydantic', client_obj_constructor)

Props = list[str]
Attrs = dict[str, Union[str, int, float]]
props_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=list)
attrs_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=dict)

class DBBase(sqlmodel.SQLModel, SpecBase):
    id: uuid.UUID = sqlmodel.Field(primary_key=True)

    def clear(self, backend, ghost=True):
        return

    def validated_with_backend(self, backend):
        for name, field in self.model_fields.items():
            # print(field.annotation, self[name], type(self[name]))
            if field.annotation in (uuid.UUID, Optional[uuid.UUID]):
                # print(name, self[name])
                if name == 'id' and self[name] is None: self[name] = uuid.uuid4()
                elif isinstance(self[name], str):
                    try:
                        self[name] = uuid.UUID(self[name])
                    except ValueError as e:
                        raise ValueError(f'bad UUID string "{self[name]}"') from e
                        # print('success')
        return self

    def to_spec(self):
        self.clsspec(**self.model_dump())

    @classmethod
    def from_spec(cls, spec):
        return cls(**spec.model_dump())

class ModelFrontend:
    def __init_subclass__(cls, models, **kw):
        super().__init_subclass__(**kw)
        cls.__models__ = models
        cls.__modelspecs__ = {name: mdl.clsspec() for name, mdl in models.items()}
        add_client_model_properties(cls)

class FastapiModelBackend:
    def __init__(self):
        self.router = fastapi.APIRouter()
        route = self.router.add_api_route
        add_backend_model_property_routes(self, self.__models__, route)

    def __init_subclass__(cls, models, **kw):
        super().__init_subclass__(**kw)
        cls.__models__ = models
        cls.__modelspecs__ = {name: mdl.clsspec() for name, mdl in models.items()}
        add_backend_model_properties(cls, models)

def add_backend_model_property_routes(self, models, router):
    for model in models:
        router(f'/{model}', getattr(self, model), methods=['GET'])
        router(f'/{model}s', getattr(self, f'{model}s'), methods=['GET'])
        router(f'/n{model}s', getattr(self, f'n{model}s'), methods=['GET'])
        router(f'/create/{model}', getattr(self, f'create_{model}'), methods=['POST'])

def add_client_model_properties(clientcls):
    '''
      Generic interface for accessing models from the server. Any name or name suffixed with 's'
      that is in frontend_model, above, will get /name from the server and turn the result(s) into
      the appropriate client model type, list of such types for plural, or None.
      '''
    for _name, _cls in clientcls.__models__.items():

        def make_funcs_forcing_closure_over_cls_name(cls=_cls, name=_name):
            def new(self, **kw) -> str:
                return self.upload(cls.clsspec()(**kw))

            def count(self, **kw) -> int:
                return self.get(f'/n{name}s', **kw)

            def multi(self, **kw) -> list[cls]:
                return [cls(self, **x) for x in self.get(f'/{name}s', **kw)]

            def single(self, **kw) -> Union[cls, None]:
                result = self.get(f'/{name}', **kw)
                return cls(self, **result) if result else None

            return single, multi, count, new

        single, multi, count, new = make_funcs_forcing_closure_over_cls_name()
        setattr(clientcls, _name, single)
        setattr(clientcls, f'{_name}s', multi)
        setattr(clientcls, f'n{_name}s', count)
        setattr(clientcls, f'new{_name}', new)

def add_backend_model_properties(backendcls, models):
    '''
    Autogen getter methods. Yes, this is better than lots of boilerplate functions that must be kept
    in sync. Any name or name suffixed with 's'
    that is in clientmodels, above, will get /name from the server and turn the result(s) into
    the appropriate client model type, list of such types for plural, or None.
    for _name, _cls in backend_model.items():
    '''
    for _name, _cls in models.items():

        def make_funcs_forcing_closure_over_name_cls(name=_name, cls=_cls):
            def create(self, model: dict) -> Union[str, int]:
                # model = cls.parse_obj(model)
                if isinstance(model, dict): model = cls(**model)
                return self.validate_and_add_to_db(model)

            def new(self, **kw) -> Union[str, int]:
                for k, v in kw.copy().items():
                    if k in models:
                        kw[f'{k}id'] = v.id
                        del kw[k]
                model = models[name](**kw)
                newid = getattr(self, f'create_{name}')(model)
                return getattr(self, f'i{name}')(newid, _ghost=True)

            def count(self, kw=None, request: fastapi.Request = None, response_model=int):
                # print('route', name, cls, kw, request, flush=True)
                if request: return self.select(cls, _count=True, **request.query_params)
                elif kw: return self.select(cls, _count=True, **kw)
                else: return self.select(cls, _count=True)

            def multi(self, kw=None, request: fastapi.Request = None, response_model=list[cls]):
                # print('route', name, cls, kw, request, flush=True)
                if request: return self.select(cls, **request.query_params)
                elif kw: return self.select(cls, **kw)
                else: return self.select(cls)

            def single(self, kw=None, request: fastapi.Request = None, response_model=Optional[cls]):
                # print('route', name, cls, kw, request, flush=True)
                if request: return self.select(cls, _single=True, **request.query_params)
                elif kw: return self.select(cls, _single=True, **kw)
                else: return self.select(cls, _single=True)

            def singleid(self, id: str, **kw) -> Union[cls, None]:
                assert id
                return self.select(cls, id=id, _single=True, **kw)

            return multi, single, singleid, count, create, new

        multi, single, singleid, count, create, new = make_funcs_forcing_closure_over_name_cls()
        setattr(backendcls, _name, single)
        setattr(backendcls, f'i{_name}', singleid)
        setattr(backendcls, f'{_name}s', multi)
        setattr(backendcls, f'n{_name}s', count)
        setattr(backendcls, f'new{_name}', new)
        if not hasattr(backendcls, f'create_{_name}'):
            setattr(backendcls, f'create_{_name}', create)
