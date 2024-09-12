import os

__all__ = ("Bunch", "bunchify", "unbunchify")

class Bunch(dict):
    def __init__(
        self,
        __arg_or_ns=None,
        _strict="__NOT_STRICT",
        _default="__NODEFALT",
        **kw,
    ):
        if __arg_or_ns is not None:
            try:
                super().__init__(__arg_or_ns)
            except TypeError:
                super().__init__(vars(__arg_or_ns))
        self.update(kw)
        self.__dict__["_special"] = dict()
        self.__dict__["_special"]["strict_lookup"] = _strict is True or _strict == "__STRICT"
        if _default == "__NODEFALT":
            _default = None
        elif _strict == "__STRICT":
            # if _default passed explicitly, and strict is not, don't be strict
            self.__dict__["_special"]["strict_lookup"] = False
        self.__dict__["_special"]["default"] = _default
        # self._clear = self.clear()
        # del self.__dict__['clear']
        for k in self:
            if hasattr(super(), k):
                raise ValueError(f"{k} is a reseved name for Bunch")

    def default(self):
        dflt = self._special["default"]
        if hasattr(dflt, "__call__"):
            return dflt()
        else:
            return dflt

    def __str__(self):
        s = "Bunch("
        s += ", ".join([f"{k}={v}" for k, v in self.items()])

        s += ")"
        if len(s) > 120:
            s = "Bunch(" + os.linesep
            if len(self) == 0:
                return "Bunch()"
            w = int(min(40, max(len(str(k)) for k in self)))
            for k, v in self.items():
                s += f'  {k:{f"{w}"}} = {v}' + os.linesep
            s += ")"
        return s

    def printme(self):
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
            s = "Bunch(" + os.linesep
            if len(self) == 0:
                return "Bunch()"
            w = int(min(40, max(len(str(k)) for k in self)))
            for k, v in self.items():
                s += f'  {k:{f"{w}"}} = {short(v)}' + os.linesep
            s += ")"
        print(s, flush=True)
        return s

    def reduce(self, func, strict=True):
        "reduce all contained iterables using <func>"
        for k in self:
            try:
                self[k] = func(self[k])
            except TypeError as ex:
                if not strict:
                    raise ex
        return self

    def accumulate(self, other, strict=True):
        "accumulate all keys in other, adding empty lists if k not in self, extend other[k] is list"
        if isinstance(other, list):
            for b in other:
                self.accumulate(b)
            return self
        if not isinstance(other, dict):
            raise TypeError("Bunch.accumulate needs Bunch or dict type")
        not_empty = len(self)
        for k in other:
            if not k in self:
                if strict and not_empty:
                    raise ValueError(f"{k} not in this Bunch")
                self[k] = list()
            if not isinstance(self[k], list):
                self[k] = [self[k]]
            o = other[k]
            if not isinstance(o, list):
                o = [o]
                self[k].extend(o)
        return self

    def __contains__(self, k):
        if k == "_special":
            return False
        try:
            return dict.__contains__(self, k) or k in self.__dict__
        except:
            return False

    def is_strict(self):
        return self._special["strict_lookup"]

    def __getitem__(self, key):
        return self.__getattr__(key)

    # try:
    # super().__getitem__(key)
    # except KeyError:
    # return self._special('default')()

    def __getattr__(self, k):
        if k == "_special":
            raise ValueError(f"_special is a reseved name for Bunch")
        if k == "__deepcopy__":
            return None
        if self._special["strict_lookup"] and not k in self:
            raise KeyError(f"Bunch is missing value for key {k}")
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return super().__getitem__(k)
            except KeyError as e:
                if self._special["strict_lookup"]:
                    raise e
                else:
                    return self.default()

    def __setattr__(self, k, v):
        if hasattr(super(), k):
            raise ValueError(f"{k} is a reseved name for Bunch")
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except KeyError:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)

    def __delattr__(self, k):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
            except KeyError:
                raise AttributeError(k)
        else:
            object.__delattr__(self, k)

    def copy(self):
        return Bunch.from_dict(super().copy())

    def set_if_missing(self, k, v):
        if k not in self:
            self[k] = v

    def sub(self, __BUNCH_SUB_ITEMS=None, _onlynone=False, exclude=[], **kw):
        if len(kw) == 0:
            if isinstance(__BUNCH_SUB_ITEMS, dict):
                kw = __BUNCH_SUB_ITEMS
            else:
                kw = vars(__BUNCH_SUB_ITEMS)
        newbunch = self.copy()
        newbunch._special = self._special
        for k, v in kw.items():
            if v is None and k in newbunch:
                del newbunch[k]
            elif not _onlynone or k not in self or self[k] is None:
                if k not in exclude:
                    newbunch.__setattr__(k, v)
        return newbunch

    def only(self, keys):
        newbunch = Bunch()
        newbunch._special = self._special
        for k in keys:
            if k in self:
                newbunch[k] = self[k]
        return newbunch

    def without(self, *dropkeys):
        newbunch = Bunch()
        newbunch._special = self._special
        for k in self.keys():
            if not k in dropkeys:
                newbunch[k] = self[k]
        return newbunch

    def visit_remove_if(self, func, recurse=True, depth=0):
        toremove = list()
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
        args = ", ".join(["%s=%r" % (key, self[key]) for key in self.keys()])
        return "%s(%s)" % (self.__class__.__name__, args)

    def asdict(self):
        return unbunchify(self)

    @staticmethod
    def from_dict(d):
        return bunchify(d)

def bunchify(x):
    if isinstance(x, dict):
        return Bunch(**x)
    elif isinstance(x, (list, tuple)):
        return type(x)(bunchify(v) for v in x)
    else:
        return x

def unbunchify(x):
    if isinstance(x, dict):
        return dict((k, unbunchify(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(unbunchify(v) for v in x)
    else:
        return x
