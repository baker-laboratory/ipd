import typing as t
import contextlib
import hashlib
import os
import shutil
from rapidfuzz import fuzz
from pathlib import Path
from typing import Generic, TypeVar, Mapping, Iterable
from evn.decon.item_wise import item_wise_operations
from evn.decon.attr_access import subscriptable_for_attributes
from evn._prelude.inspect import summary
from evn import NA
import evn

__all__ = ('Bunch', 'bunchify', 'unbunchify', 'make_autosave_hierarchy',
           'unmake_autosave_hierarchy')

T = TypeVar('T', bound=t.Any)


def strmatch(a, b, fuzzy=0.2, partial='auto'):
    if not fuzzy:
        return a in b
    func = fuzz.ratio
    if partial is True or (partial == 'auto' and max(len(a), len(b)) > 10):
        func = fuzz.partial_ratio
    return func(a, b) >= 1 - fuzzy


def bunchfind(haystack,
              needle,
              fuzzy=0,
              partial='auto',
              path='',
              seenit=None,
              matcher=fuzz.partial_ratio):
    seenit = seenit or set()
    found = {}
    if id(haystack) in seenit:
        return found
    seenit.add(id(haystack))
    items = enumerate(haystack)
    if isinstance(haystack, Mapping):
        found |= {
            f'{path}{k}': v
            for k, v in haystack.items() if strmatch(needle, k, fuzzy, partial)
        }
        items = haystack.items()
    for k, v in items:
        if isinstance(v, (Mapping, Iterable)):
            found |= bunchfind(v, needle, fuzzy, partial, f'{path}{k}.',
                               seenit, matcher)
    return found


@subscriptable_for_attributes
@item_wise_operations
class Bunch(dict, Generic[T]):
    """
    a dot-accessable dict subclass with defaultdict and chainmap functionallity

    keys must be strings. Can autosync with a .yaml file on disk. supports parent-child relationships. has considerable runtime overhead compared to a normal dict, so dont use in place of one. can
    """

    def __init__(
        self,
        __arg_or_ns=None,
        _strict='__STRICT',
        _default: t.Any = '__NODEFALT',
        _storedefault=True,
        _autosave=None,
        _autoreload=None,
        _parent=None,
        _split='',
        _like=None,
        _frozen=False,
        _flagmode=False,
        **kw,
    ):
        if __arg_or_ns is not None:
            try:
                super().__init__(__arg_or_ns)
            except TypeError:
                super().__init__(vars(__arg_or_ns))
        self.update(kw)

        if _default == '__NODEFALT':
            _default = None
        self.__dict__['_config'] = {}
        conf = self.__dict__['_config']
        conf['strict_lookup'] = _strict is True or _strict == '__STRICT'
        conf['default'] = _default
        conf['storedefault'] = _storedefault
        conf['autosave'] = str(_autosave) if _autosave else None
        conf['autoreload'] = _autoreload
        conf['split'] = _split
        conf['frozen'] = _frozen
        conf['flagmode'] = _flagmode
        if conf['autoreload']:
            Path(conf['autoreload']).touch()
            with open(conf['autoreload'], 'rb') as inp:
                self._conf('autoreloadhash',
                           hashlib.md5(inp.read()).hexdigest())
        conf['parent'] = _parent
        for k in list(self.keys()):
            if hasattr(super(), k):
                self[f'{k}_'] = super().__getitem__(k)
                del self[k]
                # print(f'WARNING {k} is a reserved name for dict, renaming to {k}_')
        if _like:
            conf |= _like.__dict__['_config']

    _find = bunchfind

    def _conf(self, k, v: t.Any = NA) -> t.Any:
        if '_config' not in self.__dict__:
            return ''
        if v == NA:
            return self.__dict__['_config'][k]
        else:
            self.__dict__['_config'][k] = v

    def _autoreload_check(self):
        if not self._conf('autoreload'):
            return
        import yaml

        with open(self._conf('autoreload'), 'rb') as inp:
            newhash = hashlib.md5(inp.read()).hexdigest()
        # print('_autoreload_check', newhash, self._conf('autoreloadhash'))
        if self._conf('autoreloadhash') == newhash:
            return
        self._conf('autoreloadhash', newhash)
        # disable autosave
        orig = self._conf('autosave')
        self._conf('autosave', None)
        # print('RELOAD FROM FILE', self._conf('autoreload'))
        with open(self._conf('autoreload')) as inp:
            new = yaml.load(inp, yaml.Loader)
        special = self._conf
        super().clear()
        for k, v in new.items():
            self[k] = make_autosave_hierarchy(
                v,
                _parent=(self, None),
                _default=self._conf('default'),
                _strict=self._conf('strict_lookup'))
        self._conf('autosave', orig)
        assert self._conf == special

    def _notify_changed(self, k=None, v=None):  # sourcery skip: extract-method
        if self._conf('parent'):
            parent, selfkey = self._conf('parent')
            return parent._notify_changed(f'{selfkey}.{k}', v)
        if self._conf('autosave'):
            import yaml

            if k:
                k = k.split('.')[0]
                if isinstance(v, (list, set, tuple, Bunch)):
                    self[k] = make_autosave_hierarchy(
                        self[k],
                        _parent=(self, None),
                        _default=self._conf('default'),
                        _strict=self._conf('strict_lookup'),
                    )
            os.makedirs(os.path.dirname(self._conf('autosave')), exist_ok=True)
            with open(self._conf('autosave') + '.tmp', 'w') as out:
                yaml.dump(unmake_autosave_hierarchy(self), out)
            shutil.move(
                self._conf('autosave') + '.tmp', self._conf('autosave'))
            with open(self._conf('autoreload'), 'rb') as inp:
                self._conf('autoreloadhash',
                           hashlib.md5(inp.read()).hexdigest())
                # print('SAVE TO ', self._conf('autosave'))

    def _merge(self, other, layer: str = ''):
        for key in other:
            if key in self:
                if isinstance(self[key], dict) and isinstance(
                        other[key], dict):
                    if not isinstance(self[key], Bunch):
                        self[key] = Bunch(self[key], _like=self)
                    self[key]._merge(other[key])
                else:
                    self[key] = other[key]
            else:
                self[key] = other[key]
        return self

    def default(self, key):
        default = self._conf('default')
        if default == 'bunchwithparent':
            new = Bunch(_parent=(self, None),
                        _default='bunchwithparent',
                        _strict=self._conf('strict_lookup'))
            special = new._config.copy()
            special['parent'] = id(special['parent'])
            # print('new child bunch:', key)  #, '_config:', special)
            return new
        if default == Bunch:
            return Bunch(_like=self)
        if not callable(default):
            return default
        try:
            return default()
        except TypeError:
            return default(key)

    def __eq__(self, other):
        self._autoreload_check()
        if hasattr(other, '_autoreload_check'):
            other._autoreload_check()
        return super().__eq__(other)

    def reduce(self, func, strict=True):
        "reduce all contained iterables using <func>"
        self._autoreload_check()
        for k in self:
            try:
                self[k] = func(self[k])
            except TypeError as ex:
                if not strict:
                    raise ex
        return self

    def accumulate(self, other, strict=True):
        "accumulate all keys in other, adding empty lists if k not in self, extend other[k] is list"
        if self._conf('frozen'):
            raise ValueError('Bunch is frozen')
        self._autoreload_check()
        if isinstance(other, list):
            for b in other:
                self.accumulate(b)
            return self
        if not isinstance(other, dict):
            raise TypeError('Bunch.accumulate needs Bunch or dict type')
        not_empty = len(self)
        for k in other:
            if k not in self:
                if strict and not_empty:
                    raise ValueError(f'{k} not in this Bunch')
                self[k] = []
            if not isinstance(self[k], list):
                self[k] = [self[k]]
            o = other[k]
            if not isinstance(o, list):
                o = [o]
                self[k].extend(o)
        return self

    def __contains__(self, k):
        self._autoreload_check()
        if k == '_conf':
            return False
        if self._conf('flagmode'):
            return self._contains_flagmode(k)
        return self._contains(k)

    def _contains(self, k):
        try:
            return dict.__contains__(self, k) or k in self.__dict__
        except KeyError:
            return False

    def _contains_flagmode(self, k):
        try:
            return self[k]
        except (KeyError, AttributeError):
            assert 0
            self[k] = self._default(k)
            return self[k]

    def __getattr__(self, k: str) -> T:
        self._autoreload_check()
        if k == '_config':
            raise ValueError('_config is a reseved name for Bunch')
        if k == '__deepcopy__':
            return None
        if self.__dict__['_config']['strict_lookup'] and not self._contains(k):
            if self._conf('default'):
                self.__dict__[k] = self.default(k)
                return self[k]
            raise AttributeError(f'Bunch is missing value for key {k}')
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return super().__getitem__(k)
            except KeyError as e:
                if self.__dict__['_config']['strict_lookup']:
                    raise e
                if self._conf('storedefault'):
                    self[k] = self.default(k)
                    return self[k]
                return self[k]

    def __getitem__(self, key: str) -> T:
        if not isinstance(key, str):
            return [getattr(self, k) for k in key]
        return self.__getattr__(key)

    def _get_split(self, keys, create=False):
        for split in self._conf('split'):
            if split in keys and split:
                obj = self
                for k in keys.split():
                    if create:
                        obj = obj.setdefault(k, Bunch(_like=self))
                    else:
                        obj = obj[k]
                return obj
        return self.__getattr__(keys)

    def __setitem__(self, k: str, v: T):
        if self._conf('frozen'):
            raise ValueError('Bunch is frozen')
        for split in self._conf('split'):
            if split in k and split:
                obj, keys, klast = self, *k.rsplit(split, 1)
                for k in keys.split():
                    obj[k] = obj = obj[k] if k in obj else Bunch(_like=self)
                obj[klast] = v
                return
        super().__setitem__(k, v)

    def __setattr__(self, k: str, v: T):
        if self._conf('frozen'):
            raise ValueError('Bunch is frozen')
        if hasattr(super(), k):
            raise ValueError(f'{k} is a reseved name for Bunch')
        if k.startswith('__'):
            self.__dict__[k] = v
            return
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
                self._notify_changed(k, v)
            except KeyError as e:
                raise AttributeError(k) from e
        else:
            object.__setattr__(self, k, v)
            self._notify_changed(k, v)

    def __delattr__(self, k):
        if self._conf('frozen'):
            raise ValueError('Bunch is frozen')
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
                self._notify_changed(k)
            except KeyError as e:
                raise AttributeError(k) from e
        else:
            object.__delattr__(self, k)
            self._notify_changed(k)

    # def __setitem__(self, k:str, v):
    # super().__setitem__(k, v)
    # self._notify_changed(k, v)

    def __delitem__(self, k):
        if self._conf('frozen'):
            raise ValueError('Bunch is frozen')
        super().__delitem__(k)
        self._notify_changed(k)

    def copy(self):
        self._autoreload_check()
        return Bunch.from_dict(super().copy(), _like=self)

    def set_if_missing(self, k: str, v):
        self._autoreload_check()
        if k not in self:
            self[k] = v
            self._notify_changed(k, v)

    def sub(self, __BUNCH_SUB_ITEMS=None, _onlynone=False, exclude=[], **kw):
        if self._conf('frozen'):
            raise ValueError('Bunch is frozen')
        self._autoreload_check()
        if not kw:
            if isinstance(__BUNCH_SUB_ITEMS, dict):
                kw = __BUNCH_SUB_ITEMS
            else:
                kw = vars(__BUNCH_SUB_ITEMS)
        newbunch = self.copy()
        newbunch._config = self.__dict__['_config']
        for k, v in kw.items():
            if v is None and k in newbunch:
                del newbunch[k]
            elif not _onlynone or k not in self or self[k] is None:
                if k not in exclude:
                    newbunch.__setattr__(k, v)
        return newbunch

    def only(self, keys):
        self._autoreload_check()
        newbunch = Bunch()
        newbunch._config = self.__dict__['_config']
        for k in keys:
            if k in self:
                newbunch[k] = self[k]
        return newbunch

    def without(self, *dropkeys):
        self._autoreload_check()
        newbunch = Bunch()
        newbunch._config = self.__dict__['_config']
        for k in self.keys():
            if k not in dropkeys:
                newbunch[k] = self[k]
        return newbunch

    def visit_remove_if(self, func, recurse=True, depth=0):
        self._autoreload_check()
        toremove = []
        for k, v in self.__dict__.items():
            if k == '_config':
                continue
            if func(k, v, depth):
                toremove.append(k)
            elif isinstance(v, Bunch) and recurse:
                v.visit_remove_if(func, recurse, depth=depth + 1)
        for k, v in self.items():
            if func(k, v, depth):
                toremove.append(k)
            elif isinstance(v, Bunch) and recurse:
                v.visit_remove_if(func, recurse, depth=depth + 1)
        for k in toremove:
            self.__delattr__(k)

    def __add__(self, addme):
        self._autoreload_check()
        newbunch = self.copy()
        for k, v in addme.items():
            if k in self:
                newbunch.__setattr__(k, self[k] + v)
            else:
                newbunch.__setattr__(k, v)
        return newbunch

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __str__(self):
        self._autoreload_check()
        s = 'Bunch(' + ', '.join([f'{k}={v}' for k, v in self.items()])
        s += ')'
        if len(s) > 120:
            s = f'Bunch({os.linesep}'
            if len(self) == 0:
                return 'Bunch()'
            w = int(min(40, max(len(str(k)) for k in self)))
            for k, v in self.items():
                s += f'  {k:{f"{w}"}} = {summary(v)}{os.linesep}u'
            s += ')'
        return s

    def printme(self):
        self._autoreload_check()

        def short(thing):
            s = str(thing)
            if len(s) > 80:
                with contextlib.suppress(ImportError):
                    np = evn.lazyimport('numpy')

                    if isinstance(thing, np.ndarray):
                        s = f'shape {thing.shape}'
                    else:
                        s = str(s)[:67].replace('\n', '') + '...'

            return s

        s = 'Bunch(' + ', '.join([f'{k}={v}' for k, v in self.items()]) + ')'
        if len(s) > 120:
            s = f'Bunch({os.linesep}'
            if len(self) == 0:
                return 'Bunch()'
            w = int(min(40, max(len(str(k)) for k in self)))
            for k, v in self.items():
                s += f'  {k:{f"{w}"}} = {short(v)}{os.linesep}'
            s += ')'
        print(s, flush=True)
        return s

    def __repr__(self):
        self._autoreload_check()
        # args = ["%s=%r" % (k, v) for k, v in self.items()]
        # args = str.join(',\n  ', args)
        # return rf"{self.__class__.__name__}(\n  {args})"
        return str(self)

    def asdict(self):
        return unbunchify(self)

    @staticmethod
    def from_dict(d, _like=None):
        return bunchify(d, _like=_like)


class BunchChild:

    def __init__(self, *a, _parent, **kw):
        super().__init__(*a, **kw)
        assert isinstance(_parent[0], Bunch)
        self._parent = _parent

    # def __str__(self):
    #    return f'{self.__class__.__name__}<{super().__str__()}>'

    # def __repr__(self):
    #    return f'{self.__class__.__name__}<{super().__repr__()}>'


class BunchChildList(BunchChild, list):

    def append(self, elmnt):
        super().append(elmnt)
        self._parent[0]._notify_changed(self._parent[1], elmnt)

    def __setitem__(self, index, elem):
        super().__setitem__(index, elem)
        self._parent[0]._notify_changed(f'{self._parent[1]}[{index}]', elem)


class BunchChildSet(BunchChild, set):

    def add(self, elem):
        super().add(elem)
        self._parent[0]._notify_changed(self._parent[1], elem)

    def remove(self, elem):
        super().remove(elem)
        self._parent[0]._notify_changed(self._parent[1], elem)


@t.overload
def bunchify(data: dict[str, t.Any], _like: t.Optional[Bunch] = None) -> Bunch:
    ...


@t.overload
def bunchify(data: T, _like: t.Optional[Bunch]) -> T:
    ...


def bunchify(data: t.Any, _like=None):
    if isinstance(data, dict):
        return Bunch(_like=_like,
                     **{
                         k: bunchify(v, _like=_like)
                         for k, v in data.items()
                     })  # type: ignore
    elif isinstance(data, (list, tuple)):
        return type(data)(bunchify(v, _like=_like) for v in data)
    else:
        return data


def make_autosave_hierarchy(x,
                            _parent=None,
                            seenit=None,
                            _strict=True,
                            _autosave=None,
                            _default=None):
    seenit = seenit or set()
    assert id(x) not in seenit, 'x must be a Tree'
    kw = dict(seenit=seenit,
              _parent=_parent,
              _default=_default,
              _strict=_strict)
    assert _parent is None or isinstance(_parent[0], Bunch)
    if isinstance(x, dict):
        x = Bunch(**x,
                  _parent=_parent,
                  _autosave=_autosave,
                  _autoreload=_autosave,
                  _default=_default,
                  _strict=_strict)
        for k, v in x.items():
            kw['_parent'] = (x, k)
            x[k] = make_autosave_hierarchy(v, **kw)
    elif isinstance(x, list):
        val = (make_autosave_hierarchy(v, **kw) for v in x)
        x = BunchChildList(val, _parent=_parent)
    elif isinstance(x, set):
        val = (make_autosave_hierarchy(v, **kw) for v in x)
        x = BunchChildSet(val, _parent=_parent)
    elif isinstance(x, (tuple, )):
        x = type(x)(make_autosave_hierarchy(v, **kw) for v in x)
    seenit.add(id(x))
    return x


def unmake_autosave_hierarchy(x,
                              seenit=None,
                              depth=0,
                              verbose=False,
                              _autosave=None):
    seenit = seenit or set()
    assert id(x) not in seenit, 'x must be a Tree'
    kw = dict(seenit=seenit, depth=depth + 1, verbose=verbose)
    if isinstance(x, dict):
        x = dict(**x)
        for k, v in x.items():
            x[k] = unmake_autosave_hierarchy(v, **kw)
    elif isinstance(x, list):
        x = [unmake_autosave_hierarchy(v, **kw) for v in x]
    elif isinstance(x, set):
        x = {unmake_autosave_hierarchy(v, **kw) for v in x}
    elif isinstance(x, (tuple, )):
        x = type(x)(unmake_autosave_hierarchy(v, **kw) for v in x)
    seenit.add(id(x))
    return x


def unbunchify(x):
    if isinstance(x, dict):
        return {k: unbunchify(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(unbunchify(v) for v in x)
    else:
        return x
