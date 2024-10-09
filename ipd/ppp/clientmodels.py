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

def clientprop(name):
    @property
    # @functools.lru_cache
    def getter(self):
        kind, attr = name.split('.')
        val = self._pppclient.getattr(kind, self.id, attr)
        attr = attr.title()
        if attr in globals(): cls = globals()[attr]
        elif attr[:-1] in globals(): cls = globals()[attr[:-1]]
        else: raise ValueError(f'unknown type {attr}')
        if isinstance(val, list):
            return [cls(self._pppclient, **kw) for kw in val]
        return cls(self._pppclient, **val)

    return getter

class Poll(ClientMixin, PollSpec):
    id: int
    files = clientprop('poll.pollfiles')
    reviews = clientprop('poll.reviews')
    workflow = clientprop('poll.workflow')
    user = clientprop('poll.user')

class PollFile(ClientMixin, PollFileSpec):
    id: int
    poll = clientprop('file.poll')
    reviews = clientprop('file.reviews')

class Review(ClientMixin, ReviewSpec):
    id: int
    poll = clientprop('review.poll')
    file = clientprop('review.pollfile')
    workflow = clientprop('review.workflow')
    steps = clientprop('review.steps')
    user = clientprop('review.user')

class ReviewStep(ClientMixin, ReviewStepSpec):
    id: int
    review = clientprop('reviewstep.review')
    flowstep = clientprop('reviewstep.flowstep')

class PymolCMD(ClientMixin, PymolCMDSpec):
    id: int
    flowsteps = clientprop('pymolcmd.flowsteps')
    user = clientprop('pymolcmd.user')

class FlowStep(ClientMixin, FlowStepSpec):
    id: int
    cmds = clientprop('flowstep.cmds')
    workflow = clientprop('flowstep.workflow')
    reviews = clientprop('flowstep.reviews')

class Workflow(ClientMixin, WorkflowSpec):
    id: int
    steps = clientprop('workflow.steps')
    polls = clientprop('workflow.polls')
    reviews = clientprop('workflow.reviews')
    user = clientprop('workflow.user')

class User(ClientMixin, UserSpec):
    id: int
    followers = clientprop('user.followers')
    following = clientprop('user.following')
    groups = clientprop('user.groups')
    polls = clientprop('user.polls')
    reviews = clientprop('user.reviews')
    pymolcmds = clientprop('user.pymolcmds')
    workflows = clientprop('user.workflows')

class Group(ClientMixin, GroupSpec):
    id: int
    user = clientprop('groupo.user')
    users = clientprop('group.users')
