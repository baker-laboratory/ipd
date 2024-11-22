import contextlib
import os

import yaml

from ipd.bunch import Bunch, make_autosave_hierarchy
from ipd.dev.state.toggle import ToggleOrSetWithMemory

toggle_or_set = ToggleOrSetWithMemory()

class StateManager:
    """Developed for the Prettier Protein Project Pymol Plugin.

    Manages contextual state.
    """
    def __init__(self, config_file, state_file, statetypes, defaults, debugnames=None):
        self._statetype = statetypes
        self._config_file, self._state_file = config_file, state_file
        self._defaults = defaults
        self._debugnames = debugnames or set('active_cmds')
        self.load()
        self.sanity_check()

    def sanity_check(self):
        assert self._conf._special['autosave']  # type: ignore
        assert self._conf._special['autoreload']  # type: ignore
        assert self._conf._special['strict_lookup']  # type: ignore
        assert self._conf._special['default'] == 'bunchwithparent'  # type: ignore
        assert self._state._special['autosave']  # type: ignore
        assert self._state._special['autoreload']  # type: ignore
        assert not self._state._special['strict_lookup']  # type: ignore
        assert self._state._special['default'] == 'bunchwithparent'  # type: ignore
        # assert self._state.polls._special['parent'] is self._state
        assert not self._state.polls._special['strict_lookup']  # type: ignore
        assert self._state.polls._special['default'] == 'bunchwithparent'  # type: ignore
        # print('state sanity check pass')

    def load(self):
        self._conf = self.read_config(
            self._config_file,
            _strict=True,
            opts=dict(shuffle=False),
            cmds={},
        )
        self._state = self.read_config(
            self._state_file,
            _strict=False,
            active_cmds=set(),
        )

    def read_config(self, fname, _strict, **kw):
        result = Bunch(**kw)
        if os.path.exists(fname):
            with open(fname) as inp:
                result |= Bunch(yaml.load(inp, yaml.CLoader))
        mahkw = dict(_strict=_strict, _autosave=fname, _default='bunchwithparent')
        return make_autosave_hierarchy(result, **mahkw)  # type: ignore

    def save(self):
        self._conf._notify_changed()  # type: ignore
        self._state._notify_changed()  # type: ignore

    def is_global_state(self, name):
        if name in self._statetype:
            return 'global' == self._statetype[name]
        return False

    def is_per_poll_state(self, name):
        if name in self._statetype:
            return 'perpoll' == self._statetype[name]
        return True

    def set_state_type(self, name, statetype):
        assert name not in self._statetype or self._statetype[name] == statetype
        self._statetype[name] = statetype

    def __contains__(self, name):
        if self.activepoll and name in self._state.polls[self.activepoll]:  # type: ignore
            return True
        if name in self._state: return True
        if name in self._conf.opts: return True  # type: ignore
        return False

    def get(self, name, *, poll=None):
        # sourcery skip: remove-redundant-if, remove-unreachable-code
        self.sanity_check()
        if not self.is_global_state(name): poll = poll or self.activepoll
        if name in self._debugnames: print(f'GET {name} global: {self.is_global_state(name)}')
        if self.is_global_state(name) or not poll:
            if name not in self._conf.opts and name in self._defaults:  # type: ignore
                if name in self._debugnames: print(f'set default {name} to self._conf.opts')
                setattr(self._conf.opts, name, self._defaults[name])  # type: ignore
            if name not in self._conf.opts:  # type: ignore
                if name in self._debugnames: print(f'get {name} from self._state')
                return self._state[name]  # type: ignore
            if name in self._debugnames: print(f'get {name} from self._conf.opts')
            return self._conf.opts[name]  # type: ignore
        assert self.is_per_poll_state(name)
        if name not in self._state.polls[poll]:  # type: ignore
            if name in self._conf.opts:  # type: ignore
                if name in self._debugnames: print(f'get {name} set from conf.opts')
                setattr(self._state.polls[poll], name, self._conf.opts[name])  # type: ignore
            elif name in self._defaults:
                if name in self._debugnames: print(f'get {name} set from default')
                setattr(self._state.polls[poll], name, self._defaults[name])  # type: ignore
            elif name in self._debugnames:
                print(f'no attribute {name} associated with poll {poll}')
        if name in self._state.polls[poll]:  # type: ignore
            if name in self._debugnames: print(f'get {name} from perpoll')
            return self._state.polls[poll][name]  # type: ignore
        if name in self._debugnames: print(f'get {name} not found')
        return None

    def set(self, name, val, *, poll=None):
        self.sanity_check()
        if not self.is_global_state(name): poll = poll or self.activepoll
        if self.is_global_state(name) or not poll:
            with contextlib.suppress(ValueError):
                self.get(name)
            if name in self._conf.opts:  # type: ignore
                if name in self._debugnames: print(f'set {name} in self._conf.opts')
                return setattr(self._conf.opts, name, val)  # type: ignore
            else:
                if name in self._debugnames: print(f'set {name} in self._state')
                return setattr(self._state, name, val)
        if not poll:
            raise AttributeError(f'cant set per-poll attribute {name} with no active poll')
        if name in self._debugnames: print(f'set {name} perpoll to {val}')
        try:
            setattr(self._state.polls[poll], name, val)  # type: ignore
        except AttributeError as e:
            print(self.polls._special)  # type: ignore
            raise e

    __getitem__ = get
    __getattr__ = get
    __setitem__ = set

    def __setattr__(self, k, v):
        if k[0] == '_': super().__setattr__(k, v)
        else: self.set(k, v)
