import contextlib
import inspect
import ipd
import os
import pydantic
import requests
import rich
import uuid
import typing
import yaml
from datetime import datetime
from icecream import ic
from typing import Union, Optional, Callable, Annotated, get_type_hints, Type, Any

class ClientError(Exception):
    pass

def tojson(thing):
    if isinstance(thing, list):
        return f'[{",".join(tojson(_) for _ in thing)}]'
    if isinstance(thing, str):
        return thing
    return thing.json()

class ModelRef(type):
    def __class_getitem__(cls, T):
        outerns = inspect.currentframe().f_back.f_globals
        validator = pydantic.BeforeValidator(lambda x, y, outerns=outerns: _validate_ref(x, y, outerns))
        return Annotated[Annotated[_ModelRefType, validator], T]

class Unique(type):
    def __class_getitem__(cls, T):
        return Annotated[T, 'UNIQUE']

def layerof(modelcls):
    if modelcls.__name__.endswith('Spec'): return 'spec'
    if modelcls.__name__.startswith('DB'): return 'backend'
    else: return 'frontend'

class SpecBase(pydantic.BaseModel):
    id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
    ghost: bool = False
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    props: Union[list[str], str] = []
    attrs: Union[dict[str, Union[str, int, float]], str] = {}
    _errors: str = ''

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        layer = layerof(cls)
        if layer == 'spec':
            cls.__spec__ = cls
        else:
            specbase = cls.__spec__ = [c for c in cls.__bases__ if c.__name__.endswith('Spec')]
            cls.__spec__ = specbase[0] if specbase else None
            if cls.__spec__:
                if layer == 'backend':
                    cls.__spec__.__backend__ = cls
                else:
                    cls.__spec__.__frontend__ = cls

    def __heash__(self):
        return self.id

    def to_spec(self):
        return self.__spec__(**self.model_dump())

    @classmethod
    def from_spec(cls, spec):
        return cls(**spec.model_dump())

    @classmethod
    def kind(cls):
        return cls.__spec__.__name__.replace('Spec', '').lower()

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
            if isinstance(prop, (tuple, list)):
                print('multiple:')
            elif hasattr(prop, 'print_full'):
                prop.print_full(seenit, depth)
            else:
                print(prop)
                # for p in prop:
                # if p.id not in seenit: p.print_full(seenit, depth)

def make_client_models(spec_models, backend_models):
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
                proptype = typing.get_args(clsdb.__annotations__[propname])[0]
            else:
                continue
            if hasattr(proptype, '__origin__'):
                propkind = typing.get_args(proptype)[0]
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

class ModelFrontend:
    def __init_subclass__(cls, models, **kw):
        super().__init_subclass__(**kw)
        cls.__models__ = models
        cls.__modelspecs__ = {name: mdl.__spec__ for name, mdl in models.items()}
        _add_client_model_properties(cls)

    def getattr(self, thing, id, attr):
        return self.get(f'/getattr/{thing}/{id}/{attr}')

    def setattr(self, thing, attr, val):
        thingtype = thing.__class__.__name__.lower()
        return self.post(f'/setattr/{thingtype}/{thing.id}/{attr}', val)

    def get(self, url, **kw):
        ipd.ppp.fix_label_case(kw)
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        if not self.testclient: url = f'http://{self.server_addr}/ppp{url}'
        if self.testclient:
            return self.testclient.get(url)
        response = requests.get(url)
        if response.status_code != 200:
            reason = response.reason if hasattr(response, 'reason') else '???'
            raise ClientError(f'GET failed URL: "{url}"\n    RESPONSE: {response}\n    '
                              f'REASON:   {reason}\n    CONTENT:  {response.content.decode()}')
        return response.json()

    def post(self, url, thing, **kw):
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        if not self.testclient: url = f'http://{self.server_addr}/ppp{url}'
        body = tojson(thing)
        # print('POST', url, type(thing), body)
        if self.testclient: response = self.testclient.post(url, content=body)
        else: response = requests.post(url, body)
        # ic(response)
        if response.status_code != 200:
            if len(str(body)) > 2048: body = f'{body[:1024]} ... {body[-1024:]}'
            reason = response.reason if hasattr(response, 'reason') else '???'
            raise ClientError(f'POST failed "{url}"\n    BODY:     {body}\n    '
                              f'RESPONSE: {response}\n    REASON:   {reason}\n    '
                              f'CONTENT:  {response.content.decode()}')
        return response.json()

    def remove(self, thing):
        assert isinstance(thing, ipd.ppp.SpecBase), f'cant remove type {thing.__class__.__name__}'
        thingname = thing.__class__.__name__.replace('Spec', '').lower()
        return self.get(f'/remove/{thingname}/{thing.id}')

    def upload(self, thing, _dispatch_on_type=True, **kw):
        if _dispatch_on_type and isinstance(thing, ipd.ppp.PollSpec): return self.upload_poll(thing)
        if _dispatch_on_type and isinstance(thing, ipd.ppp.ReviewSpec): return self.upload_review(thing)
        thing = thing.to_spec()
        # print('upload', type(thing), kw)
        if thing._errors:
            return thing._errors
        kind = type(thing).__name__.replace('Spec', '').lower()
        # ic(kind)
        result = self.post(f'/create/{kind}', thing, **kw)
        # ic(result)
        try:
            result = uuid.UUID(result)
            return ipd.ppp.client_models[kind](self, **self.get(f'/{kind}', id=result))
        except ValueError:
            return result

def _add_client_model_properties(clientcls):
    '''
      Generic interface for accessing models from the server. Any name or name suffixed with 's'
      that is in frontend_model, above, will get /name from the server and turn the result(s) into
      the appropriate client model type, list of such types for plural, or None.
      '''
    for _name, _cls in clientcls.__models__.items():

        def make_funcs_forcing_closure_over_cls_name(cls=_cls, name=_name):
            def new(self, **kw) -> str:
                return self.upload(cls.__spec__(**kw))

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

_ModelRefType = Optional[Union[uuid.UUID, str]]

def _label_field(cls):
    if hasattr(cls, '_label'): return cls._label.default
    return 'name'

def _validate_ref(val: Union[uuid.UUID, str], valinfo, spec_namespace):
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
