import functools
import ipd
from typing import Optional
import pydantic

requests = ipd.lazyimport('requests', pip=True)
rich = ipd.lazyimport('rich', 'Rich', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)

print = rich.print

class ClientMixin(pydantic.BaseModel):
    id: int
    _pppclient: Optional['PPPClient'] = None

    def __init__(self, pppclient, **kw):
        super().__init__(**kw)
        self._pppclient = pppclient

    def __hash__(self):
        return self.id

    def _validated(self):
        'noop, as validation should have happened at Spec stage'
        return self

    def __setattr__(self, name, val):
        assert name != 'id', 'cant set id via client'
        if self._pppclient: return self._pppclient.setattr(self, name, val)
        super().__setattr__(name, val)

def client_obj_representer(dumper, obj):
    data = obj.dict()
    data['class'] = obj.__class__.__name__
    return dumper.represent_scalar('!Pydantic', data)

def client_obj_constructor(loader, node):
    value = loader.construct_scalar(node)
    cls = globals()[value.pop('class')]
    return cls(**value)

yaml.add_representer(ClientMixin, client_obj_representer)
yaml.add_constructor('!Pydantic', client_obj_constructor)

class Poll(ClientMixin, ipd.ppp.PollSpec):
    _dbprops: tuple[str] = ('pollfiles', 'reviews', 'workflow', 'user')

class FileKind(ClientMixin, ipd.ppp.FileKindSpec):
    _dbprops: tuple[str] = ('pollfiles')

class PollFile(ClientMixin, ipd.ppp.PollFileSpec):
    _dbprops: tuple[str] = ('poll', 'reviews', 'filekind', 'parent', 'children')

class Review(ClientMixin, ipd.ppp.ReviewSpec):
    _dbprops: tuple[str] = ('poll', 'pollfile', 'workflow', 'steps', 'user')

class ReviewStep(ClientMixin, ipd.ppp.ReviewStepSpec):
    _dbprops: tuple[str] = ('review', 'flowstep')

class PymolCMD(ClientMixin, ipd.ppp.PymolCMDSpec):
    _dbprops: tuple[str] = ('flowsteps', 'user')

class FlowStep(ClientMixin, ipd.ppp.FlowStepSpec):
    _dbprops: tuple[str] = ('pymolcmds', 'workflow', 'reviews')

class Workflow(ClientMixin, ipd.ppp.WorkflowSpec):
    _dbprops: tuple[str] = ('steps', 'polls', 'reviews', 'user')

class User(ClientMixin, ipd.ppp.UserSpec):
    _dbprops: tuple[str] = ('followers', 'following', 'groups', 'polls', 'reviews', 'pymolcmds', 'workflows')

class Group(ClientMixin, ipd.ppp.GroupSpec):
    _dbprops: tuple[str] = ('user', 'users')

frontend_model = {name: globals()[spec.__name__[:-4]] for name, spec in ipd.ppp.spec_model.items()}

def clientprop(name):
    @property
    # @functools.lru_cache
    def getter(self):
        kind, attr = name.split('.')
        val = self._pppclient.getattr(kind, self.id, attr)
        if val is None: raise AttributeError(f'kind {kind} id {self.id} attr {attr} is None')
        if attr in frontend_model: cls = frontend_model[attr]
        elif attr.rstrip('s') in frontend_model: cls = frontend_model[attr[:-1]]
        else: raise ValueError(f'unknown type {attr}')
        if isinstance(val, list):
            return tuple(cls(self._pppclient, **kw) for kw in val)
        return cls(self._pppclient, **val)

    return getter

for cls in frontend_model.values():
    name = cls.__name__.lower()
    for prop in cls._dbprops.default:
        setattr(cls, prop, clientprop(f'{name}.{prop}'))

    def make_funcs_forcing_closure_over_clsspec(clsspec=ipd.ppp.spec_model[name]):
        def spec(self):
            return clsspec(**self.dict())

        return spec

    cls.spec = make_funcs_forcing_closure_over_clsspec()
