import ipd

_manager = None

class RedoThisDesign(Exception):
    pass

class SieveError(ValueError):
    pass

class SieveManager:
    def __init__(self):
        self._sieve_classes = {}
        self.sieves = []

    def init_sieves(self, conf, opt):
        self.conf = conf
        self.opt = opt
        self.opt.enabled = True  # default
        if 'sieve' not in conf: return
        for key in conf['sieve']:
            if key[0].isupper():  # assume is class nam
                if key not in self._sieve_classes:
                    err = f'unknown sieve type {key}, available are: [{", ".join(self._sieve_classes.keys())}]'
                    raise SieveError(err)
                cls = self._sieve_classes[key]
                # print(f'CREATE SIEVE {cls} {cls.__name__}')
                instance = cls(manager=self, conf=conf['sieve'][cls.__name__])
                self.sieves.append(instance)
            else:  # assume is parameter
                self.opt.parse_dynamic_param(key, conf['sieve'][key], overwrite=True)

    def apply_sieves(self, t, **kw):
        if not self.sieves: return
        if not self.opt.enabled: return
        cache = {}
        progress = 1.0 - t / self.conf.diffuser.T
        for sieve in self.sieves:
            if not sieve(progress=progress, cache=cache, **kw):
                print(f'Sieve {sieve.__class__.__name__} FAIL, restarting design')
                raise RedoThisDesign()

_manager = SieveManager()

def create_sieve_manager(conf: 'omegaconf.OmegaConf', opt: 'ipd.dev.DynamicParameters'):  # type: ignore
    _manager.init_sieves(conf, opt)  # type: ignore

def apply(*a, **kw):
    _manager.apply_sieves(*a, **kw)  # type: ignore

class Sieve:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _manager._sieve_classes[cls.__name__] = cls  # type: ignore

    def __init__(self, manager, conf):
        self.manager = manager
        self.conf = conf
        self.opt = manager.opt.clone()
        for k, v in self.conf.items():
            self.opt.parse_dynamic_param(k, v, overwrite=True)
