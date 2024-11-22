import hashlib
import os
import shutil
from pathlib import Path
from typing import Generic, TypeVar

__all__ = ('Bunch', 'bunchify', 'unbunchify', 'make_autosave_hierarchy', 'unmake_autosave_hierarchy')

T = TypeVar('T')

class Bunch(dict, Generic[T]):
    def __init__(
        self,
        __arg_or_ns=None,
        _strict='__STRICT',
        _default='__NODEFALT',
        _storedefault=True,
        _autosave=None,
        _autoreload=None,
        _parent=None,
        **kw,
    ):
        if __arg_or_ns is not None:
            try:
                super().__init__(__arg_or_ns)
            except TypeError:
                super().__init__(vars(__arg_or_ns))
        self.update(kw)
        if _default == "__NODEFALT": _default = None
        self.__dict__['_special'] = {}
        self.__dict__["_special"]['strict_lookup'] = _strict is True or _strict == "__STRICT"
        self.__dict__["_special"]['default'] = _default
        self.__dict__["_special"]["storedefault"] = _storedefault
        self.__dict__["_special"]["autosave"] = str(_autosave) if _autosave else None
        self.__dict__["_special"]["autoreload"] = _autoreload
        if _autoreload:
            Path(_autoreload).touch()
            self.__dict__['_special']['autoreloadhash'] = hashlib.md5(open(_autoreload, 'rb').read()).hexdigest()
        self.__dict__["_special"]["parent"] = _parent
        for k in self:
            if hasattr(super(), k):
                raise ValueError(f"{k} is a reseved name for Bunch")

    def _autoreload_check(self):
        if not self.__dict__['_special']['autoreload']: return
        import yaml
        with open(self.__dict__['_special']['autoreload'], 'rb') as inp:
            newhash = hashlib.md5(inp.read()).hexdigest()
        # print('_autoreload_check', newhash, self.__dict__['_special']['autoreloadhash'])
        if self.__dict__['_special']['autoreloadhash'] == newhash: return
        self.__dict__['_special']['autoreloadhash'] = newhash
        # disable autosave
        orig, self.__dict__['_special']['autosave'] = self.__dict__['_special']['autosave'], None
        # print('RELOAD FROM FILE', self.__dict__['_special']['autoreload'])
        with open(self.__dict__['_special']['autoreload']) as inp:
            new = yaml.load(inp, yaml.Loader)
        special = self._special
        super().clear()
        for k, v in new.items():
            self[k] = make_autosave_hierarchy(  # type: ignore
                v,  # type: ignore
                _parent=(self, None),
                _default=self._special['default'],  # type: ignore
                _strict=self._special['strict_lookup'])  # type: ignore
        self.__dict__['_special']['autosave'] = orig
        assert self._special == special

    def _notify_changed(self, k=None, v=None):  # sourcery skip: extract-method
        if self._special['parent']:  # type: ignore
            parent, selfkey = self._special['parent']  # type: ignore
            return parent._notify_changed(f'{selfkey}.{k}', v)
        if self._special['autosave']:  # type: ignore
            ic(self._special['autosave'])  # type: ignore
            import yaml
            if k:
                k = k.split('.')[0]
                if isinstance(v, (list, set, tuple, Bunch)):
                    self[k] = make_autosave_hierarchy(  # type: ignore
                        self[k],  # type: ignore
                        _parent=(self, None),
                        _default=self._special['default'],  # type: ignore
                        _strict=self._special['strict_lookup'])  # type: ignore
            os.makedirs(os.path.dirname(self._special['autosave']), exist_ok=True)  # type: ignore
            with open(self._special['autosave'] + '.tmp', 'w') as out:  # type: ignore
                yaml.dump(unmake_autosave_hierarchy(self), out)
            shutil.move(self._special['autosave'] + '.tmp', self._special['autosave'])  # type: ignore
            with open(self._special['autoreload'], 'rb') as inp:  # type: ignore
                self._special['autoreloadhash'] = hashlib.md5(inp.read()).hexdigest()  # type: ignore
                # print('SAVE TO ', self.__dict__['_special']['autosave'])

    def default(self, key):
        dflt = self.__dict__['_special']['default']
        if dflt == 'bunchwithparent':
            new = Bunch(_parent=(self, None),
                        _default='bunchwithparent',
                        _strict=self.__dict__['_special']['strict_lookup'])
            special = new._special.copy()
            special['parent'] = id(special['parent'])
            # print('new child bunch:', key)  #, '_special:', special)
            return new
        if hasattr(dflt, "__call__"):
            return dflt()
        else:
            return dflt

    def __str__(self):
        self._autoreload_check()
        s = "Bunch(" + ", ".join([f"{k}={v}" for k, v in self.items()])
        s += ")"
        if len(s) > 120:
            s = f"Bunch({os.linesep}"
            if len(self) == 0:
                return "Bunch()"
            w = int(min(40, max(len(str(k)) for k in self)))
            for k, v in self.items():
                s += f'  {k:{f"{w}"}} = {v}{os.linesep}'
            s += ")"
        return s

    def __eq__(self, other):
        self._autoreload_check()
        if hasattr(other, '_autoreload_check'): other._autoreload_check()
        return super().__eq__(other)

    def printme(self):
        self._autoreload_check()

        def short(thing):
            s = str(thing)
            if len(s) > 80:
                import numpy as np

                if isinstance(thing, np.ndarray):
                    s = f"shape {thing.shape}"
                else:
                    s = str(s)[:67].replace("\n", "") + "..."
            return s

        s = "Bunch("
        s += ", ".join([f"{k}={v}" for k, v in self.items()])

        s += ")"
        if len(s) > 120:
            s = f"Bunch({os.linesep}"
            if len(self) == 0:
                return "Bunch()"
            w = int(min(40, max(len(str(k)) for k in self)))
            for k, v in self.items():
                s += f'  {k:{f"{w}"}} = {short(v)}{os.linesep}'
            s += ")"
        print(s, flush=True)
        return s

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
        self._autoreload_check()
        if isinstance(other, list):
            for b in other:
                self.accumulate(b)
            return self
        if not isinstance(other, dict):
            raise TypeError("Bunch.accumulate needs Bunch or dict type")
        not_empty = len(self)
        for k in other:
            if k not in self:
                if strict and not_empty:
                    raise ValueError(f"{k} not in this Bunch")
                self[k] = []  # type: ignore
            if not isinstance(self[k], list):
                self[k] = [self[k]]  # type: ignore
            o = other[k]
            if not isinstance(o, list):
                o = [o]
                self[k].extend(o)  # type: ignore
        return self

    def __contains__(self, k):
        self._autoreload_check()
        if k == "_special":
            return False
        try:
            return dict.__contains__(self, k) or k in self.__dict__
        except:  # noqa
            return False

    def is_strict(self):
        return self.__dict__['_special']["strict_lookup"]

    def __getitem__(self, key: str) -> T:
        return self.__getattr__(key)

    # try:
    # super().__getitem__(key)
    # except KeyError:
    # return self.__dict__['_special']('default')()

    def __getattr__(self, k: str) -> T:
        self._autoreload_check()
        if k == "_special":
            raise ValueError("_special is a reseved name for Bunch")
        if k == "__deepcopy__":
            return None  # type: ignore
        if self.__dict__['_special']["strict_lookup"] and k not in self:
            raise AttributeError(f"Bunch is missing value for key {k}")
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return super().__getitem__(k)
            except KeyError as e:
                if self.__dict__['_special']["strict_lookup"]:
                    raise e
                if self._special['storedefault']:  # type: ignore
                    self[k] = self.default(k)  # type: ignore
                    return self[k]
                return self[k]

    def __setitem__(self, k: str, v: T):
        # if k == 'polls':
        # print('set polls trace:')
        # traceback.print_stack()
        super().__setitem__(k, v)

    def __setattr__(self, k: str, v: T):
        assert k != 'polls'
        if hasattr(super(), k):
            raise ValueError(f"{k} is a reseved name for Bunch")
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
        super().__delitem__(k)
        self._notify_changed(k)

    def copy(self):  # type: ignore
        self._autoreload_check()
        return Bunch.from_dict(super().copy())

    def set_if_missing(self, k: str, v):
        self._autoreload_check()
        if k not in self:
            self[k] = v
            self._notify_changed(k, v)

    def sub(self, __BUNCH_SUB_ITEMS=None, _onlynone=False, exclude=[], **kw):
        self._autoreload_check()
        if not kw:
            if isinstance(__BUNCH_SUB_ITEMS, dict):
                kw = __BUNCH_SUB_ITEMS
            else:
                kw = vars(__BUNCH_SUB_ITEMS)
        newbunch = self.copy()
        newbunch._special = self.__dict__['_special']  # type: ignore
        for k, v in kw.items():
            if v is None and k in newbunch:
                del newbunch[k]  # type: ignore
            elif not _onlynone or k not in self or self[k] is None:
                if k not in exclude:
                    newbunch.__setattr__(k, v)
        return newbunch

    def only(self, keys):
        self._autoreload_check()
        newbunch = Bunch()
        newbunch._special = self.__dict__['_special']
        for k in keys:
            if k in self:
                newbunch[k] = self[k]
        return newbunch

    def without(self, *dropkeys):
        self._autoreload_check()
        newbunch = Bunch()
        newbunch._special = self.__dict__['_special']
        for k in self.keys():
            if k not in dropkeys:
                newbunch[k] = self[k]
        return newbunch

    def visit_remove_if(self, func, recurse=True, depth=0):
        self._autoreload_check()
        toremove = []
        for k, v in self.__dict__.items():
            if k == "_special":
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

    def __repr__(self):
        self._autoreload_check()
        args = ", ".join(["%s=%r" % (key, self[key]) for key in self.keys()])
        return f"{self.__class__.__name__}({args})"

    def asdict(self):
        return unbunchify(self)

    @staticmethod
    def from_dict(d):
        return bunchify(d)

class BunchChild:
    def __init__(self, *a, _parent, **kw):
        super().__init__(*a, **kw)
        assert isinstance(_parent[0], Bunch)
        self._parent = _parent

    #def __str__(self):
    #    return f'{self.__class__.__name__}<{super().__str__()}>'

    #def __repr__(self):
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

def bunchify(x):
    if isinstance(x, dict):
        return Bunch(**{k: bunchify(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)(bunchify(v) for v in x)
    else:
        return x

def make_autosave_hierarchy(x, _parent=None, seenit=None, _strict=True, _autosave=None, _default=None):
    seenit = seenit or set()
    assert id(x) not in seenit, 'x must be a Tree'
    kw = dict(seenit=seenit, _parent=_parent, _default=_default, _strict=_strict)
    assert _parent is None or isinstance(_parent[0], Bunch)
    if isinstance(x, dict):
        x = Bunch(
            **x,
            _parent=_parent,
            _autosave=_autosave,
            _autoreload=_autosave,
            _default=_default,  # type: ignore
            _strict=_strict)  # type: ignore
        for k, v in x.items():
            kw['_parent'] = (x, k)
            x[k] = make_autosave_hierarchy(v, **kw)  # type: ignore
    elif isinstance(x, list):
        val = (make_autosave_hierarchy(v, **kw) for v in x)  # type: ignore
        x = BunchChildList(val, _parent=_parent)
    elif isinstance(x, set):
        val = (make_autosave_hierarchy(v, **kw) for v in x)  # type: ignore
        x = BunchChildSet(val, _parent=_parent)
    elif isinstance(x, (tuple, )):
        x = type(x)(make_autosave_hierarchy(v, **kw) for v in x)  # type: ignore
    seenit.add(id(x))
    return x

def unmake_autosave_hierarchy(x, seenit=None, depth=0, verbose=False, _autosave=None):
    seenit = seenit or set()
    assert id(x) not in seenit, 'x must be a Tree'
    kw = dict(seenit=seenit, depth=depth + 1, verbose=verbose)
    if isinstance(x, dict):
        x = dict(**x)
        for k, v in x.items():
            x[k] = unmake_autosave_hierarchy(v, **kw)  # type: ignore
    elif isinstance(x, list):
        x = [unmake_autosave_hierarchy(v, **kw) for v in x]  # type: ignore
    elif isinstance(x, set):
        x = {unmake_autosave_hierarchy(v, **kw) for v in x}  # type: ignore
    elif isinstance(x, (tuple, )):
        x = type(x)(unmake_autosave_hierarchy(v, **kw) for v in x)  # type: ignore
    seenit.add(id(x))
    return x

def unbunchify(x):
    if isinstance(x, dict):
        return {k: unbunchify(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(unbunchify(v) for v in x)
    else:
        return x
