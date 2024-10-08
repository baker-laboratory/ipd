import os
from datetime import datetime
import functools
import tempfile
import contextlib
from subprocess import check_output
import ipd
from pathlib import Path
import gzip
import getpass
import traceback
import socket
from typing import Optional, Union
from ipd.sym.guess_symmetry import guess_symmetry, guess_sym_from_directory
from ipd.ppp.models import (PollSpec, ReviewSpec, ReviewStepSpec, FileSpec, PymolCMDSpecError, PymolCMDSpec,
                            FlowStepSpec, WorkflowSpec, UserSpec, GroupSpec, fix_label_case)
import pydantic

requests = ipd.lazyimport('requests', pip=True)
rich = ipd.lazyimport('rich', 'Rich', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)

print = rich.print

class ClientMixin(pydantic.BaseModel):
    _pppclient: Optional['PPPClient'] = None

    def __init__(self, client, **kw):
        super().__init__(**kw)
        self._pppclient = client

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
    @functools.lru_cache
    def getter(self):
        kind, attr = name.split('.')
        val = self._pppclient.getattr(kind, self.id, attr)
        attr = attr.title()
        g = globals()
        if attr in g: cls = g[attr]
        elif attr[:-1] in g: cls = g[attr[:-1]]
        else: raise ValueError(f'unknown type {attr}')
        if isinstance(val, list):
            return [cls(self._pppclient, **kw) for kw in val]
        return cls(self._pppclient, **val)

    return getter

class Poll(ClientMixin, PollSpec):
    id: int
    files = clientprop('poll.files')
    reviews = clientprop('poll.reviews')
    workflow = clientprop('poll.workflow')

class Review(ClientMixin, ReviewSpec):
    id: int
    poll = clientprop('review.poll')
    file = clientprop('review.file')
    workflow = clientprop('review.workflow')
    steps = clientprop('review.steps')

class ReviewStep(ClientMixin, ReviewStepSpec):
    id: int
    review = clientprop('reviewstep.review')
    flowstep = clientprop('reviewstep.flowstep')

class File(ClientMixin, FileSpec):
    id: int
    poll = clientprop('file.poll')
    reviews = clientprop('file.reviews')

class PymolCMD(ClientMixin, PymolCMDSpec):
    id: int
    flowsteps = clientprop('pymolcmd.flowsteps')

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

class User(ClientMixin, UserSpec):
    id: int
    followers = clientprop('user.followers')
    following = clientprop('user.following')
    groups = clientprop('user.groups')

class Group(ClientMixin, GroupSpec):
    id: int
    users = clientprop('group.users')
