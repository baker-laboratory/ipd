import contextlib
import functools
import inspect
import sys
import typing
import uuid
from datetime import datetime
from typing import Annotated, Optional, Union

import attr
import fastapi
import pydantic
import requests
import yaml

import ipd

T = typing.TypeVar('T')

class ClientError(Exception):
    pass

def _label_field(cls):
    return cls._label.default if hasattr(cls, '_label') else 'name'

_CLIENT = None

def set_client(client: 'ClientBase'):
    global _CLIENT
    _CLIENT = client

def tojson(thing):
    if isinstance(thing, list): return f'[{",".join(tojson(_) for _ in thing)}]'
    if hasattr(thing, 'model_dump_json'): return thing.model_dump_json()
    if hasattr(thing, 'json'): return thing.json()
    return str(thing)

_ModelRefType = typing.Optional[typing.Union[uuid.UUID, str]]

class ModelRef(type):
    def __class_getitem__(cls, T):
        outerns = inspect.currentframe().f_back.f_globals
        validator = pydantic.BeforeValidator(lambda x, y, outerns=outerns: process_modelref(x, y, outerns))
        if isinstance(T, tuple): T = tuple([ipd.dev.classname_or_str(T[0]), *T[1:]])
        else: T = ipd.dev.classname_or_str(T)
        return Annotated[Annotated[_ModelRefType, validator], T]

def process_modelref(val: _ModelRefType, valinfo, spec_namespace):
    assert not isinstance(val, int), 'int id is wrong, use uuid now'
    if hasattr(val, 'id'): return val.id
    with contextlib.suppress(TypeError, ValueError, AttributeError):
        return uuid.UUID(val)
    specname = valinfo.config['title']
    if not specname.endswith('Spec'): specname += 'Spec'
    cls = spec_namespace[specname]
    field = cls.model_fields[valinfo.field_name]
    typehint = field.annotation
    assert typehint == _ModelRefType, 'typehint == _ModelRefType'
    refcls = field.metadata[1]
    if isinstance(refcls, str):
        if not refcls.endswith('Spec'): refcls += 'Spec'
        refcls = spec_namespace[refcls]
    if isinstance(val, str) and _CLIENT:
        # print(val, valinfo)
        refclsname = refcls.__name__.replace('Spec', '').lower()
        if not (ref := getattr(_CLIENT, refclsname)(**{_label_field(refcls): val}, _ghost=True)):
            raise ValueError(f'unknown {refcls.__name__[:-4]} named "{val}"')
        val = ref.id
    # print(cls, refcls, val)
    return val

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

    @pydantic.model_validator(mode='before')
    def validate_base(cls, vals):
        # if isinstance(vals, uuid.UUID): vals = dict(id=vals)
        # if issubclass(cls, SpecBase) and isinstance(vals, pydantic.BaseModel):
        # vals = vals.to_spec()
        assert isinstance(vals, dict)
        if 'name' in vals: assert ipd.dev.toname(vals['name']), f'name is bad identifier {vals["name"]}'
        return vals

    def __heash__(self):
        return self.id

    def to_spec(self) -> 'SpecBase':
        if isinstance(self, SpecBase): return self
        dump = self.model_dump()
        raise NotImplementedError('need to implement id mapping and field stripping')
        for k, v in dump.copy().items():
            if k != 'id' and k.endswith('id'):
                del dump[k]
                dump[k[:-2]] = v
        return self.__spec__(**dump)

    @classmethod
    def from_spec(cls: T, spec) -> T:
        raise NotImplementedError('need to implement id mapping and field stripping')
        dump = spec.model_dump()
        return cls(**dump)

    @classmethod
    def modelkind(cls) -> str:
        if cls.__name__.endswith('Spec'): return cls.__name__.replace('Spec', '').lower()
        if cls.__name__.startswith('DB'): return cls.__name__.replace('DB', '').lower()
        return cls.__name__.lower()

    @classmethod
    def modellayer(cls) -> str:
        return layerof(cls)

    def _copy_with_newid(self) -> typing.Self:
        return self.__class__(**{**self.model_dump(), 'id': uuid.uuid4()})

    def errors(self) -> list[str]:
        return self._errors

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        return setattr(self, k, v)

    def print_full(self, seenit=None, depth=0) -> None:
        seenit = seenit or set()
        if self.id in seenit: return
        seenit.add(self.id)
        fields = list(self.model_fields)
        depth += 1
        if hasattr(self, '__remote_props__'): fields += self.__remote_props__
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

class UploadOnMutateList(ipd.dev.Instrumented, list):
    def __init__(self, thing, attr, val, attrkind=''):
        super().__init__(val)
        self.thing, self.attr, self.attrkind = thing, attr, attrkind

    def __on_change__(self, thing):
        self.thing._client.setattr(self.thing, self.attr, [str(x.id) for x in self], self.attrkind)

def make_client_models(clientcls, trimspecs, remote_props):
    spec_models = clientcls.__spec_models__
    backend_models = clientcls.__backend_models__
    client_models = {}
    for kind, spec in spec_models.items():
        trimspec = trimspecs[kind]
        clsdb = backend_models[kind]
        clsname = spec.__name__[:-4]
        body, props = {'__annotations__': {}}, {}
        for propname in remote_props[kind]:
            if propname in clsdb.model_fields:
                proptype = clsdb.model_fields[propname].annotation
                print(spec, propname)
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
        for name, member in spec.__dict__.copy().items():
            if hasattr(member, '__layer__') and member.__layer__ == 'client':
                assert callable(member)
                body[name] = member
                delattr(spec, name)
        for attr, field in spec.model_fields.copy().items():
            if attr.endswith('id'):
                optional = field.default is None
                body['__annotations__'][attr] = Optional[uuid.UUID] if optional else uuid.UUID
                if attr in trimspec.model_fields:
                    del trimspec.model_fields[attr]
        clcls = type(clsname, (ClientModelBase, trimspec), body, remote_props=props)
        # for k, v in clcls.model_fields.items():
        # print(clsname, k, v.annotation)
        clcls.__spec__ = spec
        clcls.__backend_model__ = clsdb
        spec.__frontend_model__ = clcls
        clsdb.__frontend_model__ = clcls
        client_models[kind] = clcls
        setattr(sys.modules[clientcls.__module__], clcls.__name__, clcls)

    return client_models

class ClientModelBase(pydantic.BaseModel):
    _client: 'ClientBase' = None
    __sibling_models__: dict[str, 'ClientModelBase'] = {}

    def __init_subclass__(cls, remote_props=(), siblings=(), **kw):
        super().__init_subclass__(**kw)
        if not remote_props: return
        cls.__remote_props__ = remote_props
        cls.__sibling_models__[cls.modelkind()] = cls
        for attr, kind in cls.__remote_props__.items():

            def make_client_remote_model_property_closure(_cls=cls, _attr=attr, _kind=kind):
                # print('client prop', cls.__name__, attr, kind)

                def getter(self):
                    val = self._client.getattr(_cls.modelkind(), self.id, _attr)
                    if val is None:
                        return val
                        # raise AttributeError(f'kind {_cls.modelkind()} id {self.id} attr {_attr} is None')
                    elif _kind in self.__sibling_models__:
                        attrcls = self.__sibling_models__[_kind]
                    else:
                        raise ValueError(f'unknown type {_kind}')
                    if isinstance(val, list):
                        val = (attrcls(self._client, **kw) for kw in val)
                        return UploadOnMutateList(self, _attr, val, attrkind=_kind)
                    return attrcls(self._client, **val)

                return getter

            getter = make_client_remote_model_property_closure()
            getter.__qualname__ = f'{cls.__name__}.{attr}'
            setattr(cls, attr, property(getter))

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
            attrkind = self.__remote_props__[name] if name in self.__remote_props__ else ''
            if attrkind:
                with contextlib.suppress(AssertionError):
                    val = [v.id for v in val]
            result = self._client.setattr(self, name, val, attrkind)
            assert not result, result
            if name not in self.__remote_props__:
                super().__setattr__(name, val)
        else:
            super().__setattr__(name, val)

    def __eq__(self, other):
        return self.id == other.id

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

class ClientBase:
    def __init_subclass__(cls, Backend, **kw):
        super().__init_subclass__(**kw)
        cls.__backend_models__ = Backend.__backend_models__
        cls.__spec_models__ = {name: mdl.__spec__ for name, mdl in cls.__backend_models__.items()}
        cls.__client_models__ = make_client_models(cls, Backend.__trimspecs__, Backend.__remoteprops__)
        add_basic_client_model_methods(cls)

    def __init__(self, server_addr_or_testclient):
        if isinstance(server_addr_or_testclient, str):
            self.testclient, self.server_addr = None, server_addr_or_testclient
        elif isinstance(server_addr_or_testclient, fastapi.testclient.TestClient):
            self.testclient, self.server_addr = server_addr_or_testclient, None
        set_client(self)

    def getattr(self, thing, id, attr):
        return self.get(f'/getattr/{thing}/{id}/{attr}')

    def setattr(self, thing, attr, val, attrkind=''):
        thingtype = thing.__class__.__name__.lower()
        if attrkind:
            return self.post(f'/setattr/{thingtype}/{thing.id}/{attr}/{attrkind}', val)
        else:
            return self.post(f'/setattr/{thingtype}/{thing.id}/{attr}', val)

    def preprocess_get(self, kw):
        return kw

    def get(self, url, **kw):
        kw = self.preprocess_get(kw)
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        if self.testclient:
            response = self.testclient.get(f'/api{url}')
        else:
            url = f'http://{self.server_addr}/api{url}'
            response = requests.get(url)
        if response.status_code != 200:
            reason = response.reason if hasattr(response, 'reason') else '???'
            raise ClientError(f'GET failed URL: "{url}"\n    RESPONSE: {response}\n    '
                              f'REASON:   {reason}\n    CONTENT:  {response.content.decode()}')
        return response.json()

    def post(self, url, thing, **kw):
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        body = tojson(thing)
        if self.testclient:
            url = f'/api{url}'
            response = self.testclient.post(url, content=body)
        else:
            url = f'http://{self.server_addr}/api{url}'
            response = requests.post(url, body)
        # ic(response)
        if response.status_code != 200:
            if len(str(body)) > 2048: body = f'{body[:1024]} ... {body[-1024:]}'
            reason = response.reason if hasattr(response, 'reason') else '???'
            raise ClientError(f'POST failed "{url}"\n    BODY:     {body}\n    '
                              f'RESPONSE: {response}\n    REASON:   {reason}\n    '
                              f'CONTENT:  {response.content.decode()}')
        response = response.json()
        with contextlib.suppress((TypeError, ValueError)):
            return uuid.UUID(response)
        return response

    def remove(self, thing):
        assert isinstance(thing, ipd.ppp.SpecBase), f'cant remove type {thing.__class__.__name__}'
        thingname = thing.__class__.__name__.replace('Spec', '').lower()
        return self.get(f'/remove/{thingname}/{thing.id}')

    def upload(self, thing, _dispatch_on_type=True, modelkind=None, **kw):
        modelkind = modelkind or thing.modelkind()
        if _dispatch_on_type and hasattr(self, f'upload_{modelkind}'):
            return getattr(self, f'upload_{modelkind}')(thing, **kw)
        # thing = thing.to_spec()
        # if thing._errors: return thing._errors
        result = self.post(f'/create/{modelkind}', thing, **kw)
        try:
            newthing = self.get(f'/{modelkind}', id=result)
            newthing = self.__client_models__[modelkind](self, **newthing)
            return newthing
        except ValueError:
            return result

    def make_spec(self, cls, **kw):
        if 'exe' in cls.__spec__.model_fields:
            ic(cls, kw)
            ic(cls.__spec__.model_fields)
            # for k, v in kw.copy().items():
            # if isinstance(v, pydantic.BaseModel) and hasattr(v, 'id'):
            # kw[f'{k}id'] = v.id
            # kw[k] = v.id
        return cls.__spec__(**kw)

def add_basic_client_model_methods(clientcls):
    '''
      Generic interface for accessing models from the server. Any name or name suffixed with 's'
      that is in frontend_model, above, will get /name from the server and turn the result(s) into
      the appropriate client model type, list of such types for plural, or None.
      '''
    for _name, _cls in clientcls.__client_models__.items():

        def make_basic_client_model_methods_closure(cls=_cls, name=_name):
            def new(self, **kw) -> str:
                # return self.upload(kw, modelkind=cls.modelkind())
                return self.upload(self.make_spec(cls, **kw))

            def count(self, **kw) -> int:
                return self.get(f'/n{name}s', **kw)

            def single(self, **kw) -> Union[cls, None]:
                result = self.get(f'/{name}', **kw)
                return cls(self, **result) if result else None

            def multi(self, _names=None, **kw) -> list[cls]:
                if _names: return [cls(self, **self.get(f'/{name}', name=n)) for n in _names]
                return [cls(self, **x) for x in self.get(f'/{name}s', **kw)]

            return single, multi, count, new

        single, multi, count, new = make_basic_client_model_methods_closure()
        single.__qualname__ = f'{clientcls.__name__}.{_name}'
        multi.__qualname__ = f'{clientcls.__name__}.{_name}s'
        count.__qualname__ = f'{clientcls.__name__}.n{_name}s'
        new.__qualname__ = f'{clientcls.__name__}.new{_name}'
        setattr(clientcls, _name, single)
        setattr(clientcls, f'{_name}s', multi)
        setattr(clientcls, f'n{_name}s', count)
        setattr(clientcls, f'new{_name}', new)

def model_method(func, layer):
    @functools.wraps(func)
    def wrapper(self, *a, **kw):
        err = f'{inspect.signature(func)} only valid in {layer} model, not {self.__class__.__name__}'
        assert self.modellayer() == layer, err
        func(self, *a, **kw)

    wrapper.__layer__ = layer
    return wrapper

def spec_method(func):
    return model_method(func, 'spec')

def client_method(func):
    return model_method(func, 'client')

def backend_method(func):
    return model_method(func, 'backend')
