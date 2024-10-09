import functools
import ipd
from typing import Optional
from ipd.ppp.models import (PollSpec, ReviewSpec, ReviewStepSpec, PollFileSpec, PymolCMDSpec, FlowStepSpec,
                            WorkflowSpec, UserSpec, GroupSpec)
import pydantic

requests = ipd.lazyimport('requests', pip=True)
rich = ipd.lazyimport('rich', 'Rich', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)

print = rich.print

class ClientMixin(pydantic.BaseModel):
    _pppclient: Optional['PPPClient'] = None

    def __init__(self, pppclient, **kw):
        super().__init__(**kw)
        self._pppclient = pppclient

    def __hash__(self):
        return self.id

    def _validated(self):
        'noop, as validation should have happened at Spec stage'
        return self

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

class Poll(ClientMixin, PollSpec):
    id: int
    dbprops: tuple[str] = ('pollfiles', 'reviews', 'workflow', 'user')

class PollFile(ClientMixin, PollFileSpec):
    id: int
    dbprops: tuple[str] = ('poll', 'reviews')

class Review(ClientMixin, ReviewSpec):
    id: int
    dbprops: tuple[str] = ('poll', 'pollfile', 'workflow', 'steps', 'user')

class ReviewStep(ClientMixin, ReviewStepSpec):
    id: int
    dbprops: tuple[str] = ('review', 'flowstep')

class PymolCMD(ClientMixin, PymolCMDSpec):
    id: int
    dbprops: tuple[str] = ('flowsteps', 'user')

class FlowStep(ClientMixin, FlowStepSpec):
    id: int
    dbprops: tuple[str] = ('cmds', 'workflow', 'reviews')

class Workflow(ClientMixin, WorkflowSpec):
    id: int
    dbprops: tuple[str] = ('steps', 'polls', 'reviews', 'user')

class User(ClientMixin, UserSpec):
    id: int
    dbprops: tuple[str] = ('followers', 'following', 'groups', 'polls', 'reviews', 'pymolcmds', 'workflows')

class Group(ClientMixin, GroupSpec):
    id: int
    dbprops: tuple[str] = ('user', 'users')

models = dict(poll=Poll,
              pollfile=PollFile,
              review=Review,
              reviewstep=ReviewStep,
              pymolcmd=PymolCMD,
              flowstep=FlowStep,
              workflow=Workflow,
              user=User,
              group=Group)

def clientprop(name):
    @property
    # @functools.lru_cache
    def getter(self):
        kind, attr = name.split('.')
        val = self._pppclient.getattr(kind, self.id, attr)
        if attr in models: cls = models[attr]
        elif attr.rstrip('s') in models: cls = models[attr[:-1]]
        else: raise ValueError(f'unknown type {attr}')
        if isinstance(val, list):
            return [cls(self._pppclient, **kw) for kw in val]
        return cls(self._pppclient, **val)

    return getter

for cls in (Poll, Review, ReviewStep, PollFile, PymolCMD, FlowStep, Workflow, User, Group):
    name = cls.__name__.lower()
    for prop in cls.model_fields['dbprops'].default:
        setattr(cls, prop, clientprop(f'{name}.{prop}'))
