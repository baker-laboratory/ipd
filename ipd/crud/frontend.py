import contextlib
import functools
import inspect
import os
import sys
import typing
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional, Union

import fastapi
import httpx
import pydantic
import yaml

import ipd
from ipd.dev import str_to_json, tojson

# profiler = ipd.dev.timed
profiler = lambda f: f
tojson = profiler(tojson)
str_to_json = profiler(str_to_json)

T = typing.TypeVar('T')

class ClientError(Exception):
    pass

def _label_field(cls):
    return cls._label.default if hasattr(cls, '_label') else 'name'

_CLIENT = None

def set_client(client: 'ClientBase'):
    global _CLIENT
    _CLIENT = client

_ModelRefType = typing.Optional[typing.Union[uuid.UUID, str]]

class ModelRef(type):
    def __class_getitem__(cls, T):
        outerns = inspect.currentframe().f_back.f_globals  # type: ignore
        validator = pydantic.BeforeValidator(lambda x, y, outerns=outerns: process_modelref(x, y, outerns))
        if isinstance(T, tuple): T = tuple([ipd.dev.classname_or_str(T[0]), *T[1:]])
        else: T = ipd.dev.classname_or_str(T)
        return Annotated[Annotated[_ModelRefType, validator], T]

@profiler
def process_modelref(val: _ModelRefType, valinfo, spec_namespace):
    assert not isinstance(val, int), 'int id is wrong, use uuid now'
    if hasattr(val, 'id'): return val.id  # type: ignore
    with contextlib.suppress(TypeError, ValueError, AttributeError):
        return uuid.UUID(val)  # type: ignore
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
        for k, v in dump.copy().items():  # type: ignore
            if k != 'id' and k.endswith('id'):
                del dump[k]
                dump[k[:-2]] = v
        raise NotImplementedError('need to implement id mapping and field stripping')
        # return self.__spec__(**dump)

    @classmethod
    def from_spec(cls: T, spec) -> T:
        dump = spec.model_dump()
        raise NotImplementedError('need to implement id mapping and field stripping')
        # return cls(**dump)

    @classmethod
    def modelkind(cls) -> str:
        if cls.__name__.endswith('Spec'): return cls.__name__.replace('Spec', '').lower()
        if cls.__name__.startswith('DB'): return cls.__name__.replace('DB', '').lower()
        return cls.__name__.lower()

    @classmethod
    def modellayer(cls) -> str:
        return layerof(cls)

    def _copy_with_newid(self) -> 'SpecBase':
        return self.__class__(**{**self.model_dump(), 'id': uuid.uuid4()})

    def errors(self) -> str:
        if not hasattr(self, '_errors'): return ''
        return self._errors

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        return setattr(self, k, v)

    @profiler
    def info(
        self,
        recurse=1,
        showfields='*',
        recursefields='*',
        hidefields='datecreated ghost gpus protocols path version kind required guaranteed results',
        seenit=None,
        parent=None,
        shorten=3,
    ) -> dict:
        if recurse < 0: return {}
        showall = '*' in showfields
        recurseall = '*' in recursefields
        if isinstance(recursefields, str): recursefields = set(recursefields.split())
        if isinstance(hidefields, str): hidefields = set(hidefields.split())
        seenit = seenit or set()
        # if self.id in seenit: return {}
        fields = set(self.model_fields)
        if hasattr(self, '__remote_props__'): fields |= set(self.__remote_props__)  # type: ignore
        sfields = fields if showall else fields.intersection(showfields)
        rfields = fields if recurseall else fields.intersection(recursefields)
        # print('INFO', self.__class__.__name__, recurseall, showall, len(fields), len(sfields), len(rfields))
        kw = dict(recurse=recurse - 1,
                  showfields=showfields,
                  recursefields=recursefields,
                  hidefields=hidefields,
                  seenit=seenit | {self.id},
                  parent=self.modelkind(),
                  shorten=shorten)
        d = dict(name=self.name) if hasattr(self, 'name') else {}  # type: ignore
        for attr in sorted(sfields - {'name', 'user'} - hidefields):
            if attr.endswith('id'): continue
            if parent and parent in attr: continue
            d[attr] = getattr(self, attr)
        for attr in sorted(rfields - {'name', 'user'} - hidefields):
            if attr.endswith('id'): continue
            if parent and parent in attr: continue
            prop = getattr(self, attr)
            if attr in rfields and hasattr(prop, 'info'):
                d[attr] = prop.info(**kw)
                # print(attr, d[attr])
            elif isinstance(prop, (tuple, list)):
                if len(prop) and hasattr(prop[0], 'info'): d[attr] = [p.info(**kw) for p in prop]
                else: d[attr] = prop
        for i in range(shorten):
            for k, v in d.copy().items():
                if isinstance(v, Path): print(k, v)
                if not v: del d[k]
                elif isinstance(v, Path): d[k] = str(v)
                elif not hasattr(v, '__len__'): continue
                elif len(v) == 0: del d[k]
                elif len(v) == 1 and isinstance(v, list): d[k] = next(iter(v))
                elif len(v) == 1 and isinstance(v, dict): d[k] = next(iter(v.values()))
                elif isinstance(v, list):
                    for j, u in enumerate(v):
                        if len(u) == 0: del v[j]
                        elif isinstance(u, Path): v[j] = str(u)
                        elif len(u) == 1 and isinstance(u, list): v[j] = next(iter(u))
                        elif len(u) == 1 and isinstance(u, dict): v[j] = next(iter(u.values()))

        # d = {self.__class__.__name__: d}

        return d

    @profiler
    def str_compact(self, linelen=120, strip_labels='invars outvars name'.split(), **kw):
        import compact_json
        formatter = compact_json.Formatter()
        formatter.indent_spaces = 2
        formatter.max_inline_complexity = 10
        formatter.max_inline_length = linelen
        val = self.info(**kw)
        # rich.print(val)
        text = formatter.serialize(val).replace('"', '')  # type: ignore
        for label in strip_labels:
            text = text.replace(f'{label}: ', '')
        text = f'{self.__class__.__name__}{text}'
        compact = ['']
        for i in range(6, 1, -1):
            text = text.replace(' ' * i, ' ')
        for line in text.split(os.linesep):
            if len(compact[-1]) + len(line) < linelen: compact[-1] += line.lstrip()
            else: compact.append('    ' + line)
        compact = os.linesep.join(compact)
        return compact

    def print_compact(self, **kw):
        print(self.str_compact(**kw), flush=True)

@profiler
class UploadOnMutateList(ipd.dev.Instrumented, list):
    def __init__(self, thing, attr, val, attrkind=''):
        super().__init__(val)
        self.thing, self.attr, self.attrkind = thing, attr, attrkind

    def __on_change__(self, thing):
        self.thing._client.setattr(self.thing, self.attr, [str(x.id) for x in self], self.attrkind)

@profiler
def make_client_models(clientcls, trimspecs, remote_props):
    spec_models = clientcls.__spec_models__
    backend_models = clientcls.__backend_models__
    client_models = {}
    for kind, spec in spec_models.items():
        trimspec = trimspecs[kind]
        clsdb = backend_models[kind]
        clsname = spec.__name__[:-4]
        body, props = {'__annotations__': {}}, {}
        for attr in remote_props[kind]:
            if attr in clsdb.model_fields:
                proptype = clsdb.model_fields[attr].annotation
                print(spec, attr, proptype)
                assert 0
            elif attr in clsdb.__annotations__:
                proptype = typing.get_args(clsdb.__annotations__[attr])[0]
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
            props[attr] = propkind.replace('DB', '').lower()
        for name, member in spec.__dict__.copy().items():
            if hasattr(member, '__layer__') and member.__layer__ == 'client':
                assert callable(member)
                body[name] = member  # type: ignore
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
    _client: 'ClientBase' = None  # type: ignore
    __sibling_models__: dict[str, 'ClientModelBase'] = {}

    @profiler
    def __init_subclass__(cls, remote_props=(), siblings=(), **kw):
        super().__init_subclass__(**kw)
        if not remote_props: return
        cls.__remote_props__ = remote_props
        cls.__sibling_models__[cls.modelkind()] = cls  # type: ignore
        for attr, kind in cls.__remote_props__.items():

            def make_client_remote_model_property_closure(_cls=cls, _attr=attr, _kind=kind):
                # print('client prop', cls.__name__, attr, kind)

                def getter(self):
                    val = self._client.getattr(_cls.modelkind(), self.id, _attr)  # type: ignore
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
        return self.id  # type: ignore

    def _validated(self):
        'noop, as validation should have happened at Spec stage'
        return self

    @profiler
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
        return self.id == other.id  # type: ignore

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

@profiler
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
        elif isinstance(server_addr_or_testclient, fastapi.testclient.TestClient):  # type: ignore
            self.testclient, self.server_addr = server_addr_or_testclient, None
        set_client(self)

    def getattr(self, thing, id, attr):
        result = self.get(f'/getattr/{thing}/{id}/{attr}')
        # ic(self, thing, attr, result)
        return result

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
            response = httpx.get(url)
        if response.status_code != 200:
            reason = response.reason if hasattr(response, 'reason') else '???'  # type: ignore
            raise ClientError(f'GET failed URL: "{url}"\n    RESPONSE: {response}\n    '
                              f'REASON:   {reason}\n    CONTENT:  {response.content.decode()}')
        return ipd.dev.str_to_json(response.content.decode())

    def post(self, url, thing, **kw):
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        body = ipd.dev.tojson(thing)
        if self.testclient:
            url = f'/api{url}'
            response = self.testclient.post(url, content=body)
        else:
            url = f'http://{self.server_addr}/api{url}'
            response = httpx.post(url, data=body)  # type: ignore
        # ic(response)
        if response.status_code != 200:
            if len(str(body)) > 2048: body = f'{body[:1024]} ... {body[-1024:]}'
            reason = response.reason if hasattr(response, 'reason') else '???'  # type: ignore
            raise ClientError(f'POST failed "{url}"\n    BODY:     {body}\n    '
                              f'RESPONSE: {response}\n    REASON:   {reason}\n    '
                              f'CONTENT:  {response.content.decode()}')
        response = ipd.dev.str_to_json(response.content.decode())
        with contextlib.suppress((TypeError, ValueError)):  # type: ignore
            return uuid.UUID(response)  # type: ignore
        return response

    def remove(self, thing):
        assert isinstance(thing, SpecBase), f'cant remove type {thing.__class__.__name__}'
        thingname = thing.__class__.__name__.replace('Spec', '').lower()
        return self.get(f'/remove/{thingname}/{thing.id}')

    def upload(self, thing, _dispatch_on_type=True, modelkind=None, **kw):
        modelkind = modelkind or thing.modelkind()
        remote = []
        if not isinstance(thing, SpecBase):
            thing, remote, extra = self.make_spec(modelkind, **thing)
        if err := thing.errors(): return err
        if _dispatch_on_type and hasattr(self, f'upload_{modelkind}'):
            return getattr(self, f'upload_{modelkind}')(thing, **kw)
        # thing = thing.to_spec()
        # if thing._errors: return thing._errors
        result = self.post(f'/create/{modelkind}', thing, **kw)
        try:
            newthing = self.get(f'/{modelkind}', id=result)
            newthing = self.__client_models__[modelkind](self, **newthing)
            for r in remote:
                setattr(newthing, r, thing[r])
            return newthing
        except ValueError:
            return result

    def getorupload_by_name(self, thing, modelkind=None, **kw):
        modelkind = modelkind or thing.modelkind()
        if 'name' in thing:
            # ic(thing)
            if existing := getattr(self, f'{modelkind}s')(name=thing['name']):
                # ic(len(existing))
                assert len(existing) == 1
                return existing[0]
        return self.upload(thing, modelkind=modelkind, **kw)

    def make_spec(self, modelkind, **kw):
        cls = self.__spec_models__[modelkind]
        remoteprops = set(self.__client_models__[modelkind].__remote_props__)
        remote = {k: kw[k] for k in set(kw) & set(remoteprops)}
        args = {k: kw[k] for k in (set(kw) & set(cls.model_fields)) - remoteprops}
        extra = {k: kw[k] for k in set(kw) - set(args) - remoteprops}
        for k, v in remote.copy().items():
            if isinstance(v, ClientModelBase):
                args[k] = v.id  # type: ignore
                del remote[k]
        # ic(cls, args, remote, extra)
        return cls.__spec__(**args), remote, extra

def add_basic_client_model_methods(clientcls):
    """Generic interface for accessing models from the server.

    Any name or name suffixed with 's' that is in frontend_model, above,
    will get /name from the server and turn the result(s) into the
    appropriate client model type, list of such types for plural, or
    None.
    """
    for _name, _cls in clientcls.__client_models__.items():

        def make_basic_client_model_methods_closure(cls=_cls, name=_name):
            def new(self, **kw) -> cls:  # type: ignore
                # return self.upload(kw, modelkind=cls.modelkind())
                return self.upload(kw, modelkind=cls.modelkind())

            def count(self, **kw) -> int:
                return self.get(f'/n{name}s', **kw)

            def single(self, **kw) -> cls:  # type: ignore
                result = self.get(f'/{name}', **kw)
                return cls(self, **result) if result else None

            def singleornone(self, **kw) -> Union[cls, None]:  # type: ignore
                result = self.get(f'/{name}s', **kw)
                if not result: return None
                if len(result) > 1:
                    raise ClientError(f'singleornone {len(results)}>1 rslts {name} {cls} {kw}')  # type: ignore
                return cls(self, **result[0]) if result else None

            def multi(self, _names=None, **kw) -> list[cls]:  # type: ignore
                if _names: return [cls(self, **self.get(f'/{name}', name=n)) for n in _names]
                return [cls(self, **x) for x in self.get(f'/{name}s', **kw)]

            def getornew(self, **kw) -> cls:  # type: ignore
                if thing := singleornone(self, **kw):
                    for k, v in kw.items():
                        assert thing[k] == v
                    return thing
                return new(self, **kw)

            return {
                _name: single,
                f'{_name}ornone': singleornone,
                f'{_name}s': multi,
                f'n{_name}s': count,
                f'new{_name}': new,
                f'getornew{_name}': getornew
            }

        for attr, fn in make_basic_client_model_methods_closure().items():
            fn.__qualname__ = f'{clientcls.__name__}.{attr}'
            setattr(clientcls, attr, profiler(fn))

def model_method(func, layer):
    @functools.wraps(func)
    def wrapper(self, *a, **kw):
        err = f'{inspect.signature(func)} only valid in {layer} model, not {self.__class__.__name__}'
        assert self.modellayer() == layer, err
        func(self, *a, **kw)

    wrapper.__layer__ = layer  # type: ignore
    return wrapper

def spec_method(func):
    return model_method(func, 'spec')

def client_method(func):
    return model_method(func, 'client')

def backend_method(func):
    return model_method(func, 'backend')
