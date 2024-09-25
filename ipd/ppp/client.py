import os
from datetime import datetime
import functools
import json
import tempfile
import contextlib
import ipd
import pathlib
import gzip
import getpass
import traceback
from typing import Any, Optional

pydantic = ipd.lazyimport('pydantic', pip=True)
requests = ipd.lazyimport('requests', pip=True)
rich = ipd.lazyimport('rich', 'Rich', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
wills_pymol_crap = ipd.lazyimport('wills_pymol_crap',
                                  'git+https://github.com/willsheffler/wills_pymol_crap',
                                  pip=True)
from ordered_set import OrderedSet as ordset
from rich import print

pymol = ipd.lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger')

STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
profile = ipd.dev.timed
# profile = lambda f: f

class PPPClientError(Exception):
    pass

_checkobjnum = 0

@profile
class PollSpec(pydantic.BaseModel):
    name: str
    desc: str = ''
    path: str
    user: str = getpass.getuser()
    cmdstart: str = ''
    cmdstop: str = ''
    sym: str = ''
    ligand: str = 'unknown'
    public: bool = False
    telemetry: bool = False
    workflow: str = 'manual'
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    enddate: datetime = datetime.strptime('9999-01-01T01:01:01.1', DATETIME_FORMAT)
    props: list[str] = []
    attrs: dict[str, str | int] = {}
    _errors: str = ''

    @pydantic.model_validator(mode='after')
    def _update(self):
        # sourcery skip: merge-duplicate-blocks, remove-redundant-if, set-comprehension, split-or-ifs
        if not self.name: self.name = os.path.basename(self.path)
        self.name = self.name.title()
        if not self.desc: self.desc = f"PDBs in {self.path}"
        if not self.sym or self.ligand == 'unknown':
            nchain = 1
            try:
                global _checkobjnum
                fname = next(filter(lambda s: s.endswith(STRUCTURE_FILE_SUFFIX), os.listdir(self.path)))
                pymol.cmd.set('suspend_updates', 'on')
                with contextlib.suppress(pymol.CmdException):
                    pymol.cmd.save(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
                pymol.cmd.delete('all')
                pymol.cmd.load(os.path.join(self.path, fname), f'TMP{_checkobjnum}')
                if self.ligand == 'unknown':
                    ligs = set()
                    for a in pymol.cmd.get_model(f'TMP{_checkobjnum} and HET and not resn HOH').atom:
                        ligs.add(a.resn)
                    self.ligand = ','.join(ligs)
                if not self.sym:
                    pymol.cmd.remove(f'TMP{_checkobjnum} and not name ca')
                    chains = {a.chain for a in pymol.cmd.get_model(f'TMP{_checkobjnum}').atom}
                    nchain = len(chains)
                    xyz = pymol.cmd.get_coords(f'TMP{_checkobjnum}').reshape(len(chains), -1, 3)
                    self.sym = ipd.sym.guess_symmetry(xyz)
            except ValueError as e:
                # print(os.path.join(self.path, fname))
                traceback.print_exc()
                if nchain < 4: self.sym = 'C1'
                else: self.sym = 'unknown'
            except (AttributeError, pymol.CmdException, gzip.BadGzipFile) as e:
                self._errors = f'{type(e)} {e}'
            finally:
                pymol.cmd.delete(f'TMP{_checkobjnum}')
                pymol.cmd.set('suspend_updates', 'off')
                _checkobjnum += 1
                with contextlib.suppress(pymol.CmdException):
                    pymol.cmd.load(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
                    os.remove(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
        return self

    @pydantic.validator('props')
    def _valprops(cls, v):
        return list(v)

    def __hash__(self):
        return hash(self.path)

@profile
class ReviewSpec(pydantic.BaseModel):
    polldbkey: int
    fname: str
    permafile: str = ''
    grade: str
    user: str = getpass.getuser()
    comment: str = ''
    durationsec: int = -1
    props: list[str] = []
    attrs: dict[str, str | int | float] = {}
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    _errors: str = ''

    @pydantic.validator('grade')
    def valgrade(cls, grade):
        assert grade.upper() in 'SABCDF'
        return grade.upper()

    @pydantic.validator('fname')
    def valfname(cls, fname):
        assert os.path.exists(fname)
        return os.path.abspath(fname)

@profile
class FileSpec(pydantic.BaseModel):
    polldbkey: int
    fname: str
    public: bool = True
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    props: list[str] = []
    attrs: dict[str, str | int | float] = {}
    _errors: str = ''

    @pydantic.validator('fname')
    def valfname(cls, fname):
        assert os.path.exists(fname)
        return os.path.abspath(fname)

@profile
class PymolCMDSpecError(Exception):
    def __init__(self, message, log):
        super().__init__(message + os.linesep + log)
        self.log = log

TOBJNUM = 0

@profile
class PymolCMDSpec(pydantic.BaseModel):
    name: str
    desc: str = ''
    cmdon: str
    cmdoff: str = ''
    cmdstart: str = ''
    onstart: bool = False
    public: bool = True
    user: str = getpass.getuser()
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    props: list[str] = []
    attrs: dict[str, str | int | float] = {}
    _errors: str = ''

    def errors(self):
        return self._errors

    def check_cmds(self):
        self._check_cmds_output = '-' * 80 + os.linesep + str(self) + os.linesep + '_' * 80 + os.linesep
        self._errors = ''
        objlist = pymol.cmd.get_object_list()
        global TOBJNUM
        TOBJNUM += 1
        pymol.cmd.load(ipd.testpath('pdb/tiny.pdb'), f'TEST_OBJECT{TOBJNUM}')
        self.check_cmd('cmdstart')
        self.check_cmd('cmdon')
        self.check_cmd('cmdoff')
        pymol.cmd.delete(f'TEST_OBJECT{TOBJNUM}')
        if any([
                pymol.cmd.get_object_list() != objlist,
                any(self._check_cmds_output.lower().count(err) for err in 'error unknown unrecognized'.split())
        ]):
            # raise PymolCMDSpecError('bad pymol commands', self._check_cmds_output)
            self._errors = self._check_cmds_output
        return self

    def check_cmd(self, cmdname):
        with tempfile.TemporaryDirectory() as td:
            with open(f'{td}/stdout.log', 'w') as out:
                with ipd.dev.redirect(stdout=out, stderr=out):
                    cmd = getattr(self, cmdname)
                    cmd = cmd.replace('$subject', f'TEST_OBJECT{TOBJNUM}')
                    pymol.cmd.do(cmd, echo=False, log=False)
            msg = pathlib.Path(f'{td}/stdout.log').read_text()
            msg = cmdname.upper() + os.linesep + msg + '-' * 80 + os.linesep
            self._check_cmds_output += msg
        return self

@profile
class ClientMixin(pydantic.BaseModel):
    _pppclient: Optional['PPPClient'] = None

    def __init__(self, client, **kw):
        super().__init__(**kw)
        self._pppclient = client

    def __hash__(self):
        return self.dbkey

    def __getitem__(self, key):
        return getattr(self, key)

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
        val = self._pppclient.getattr(kind, self.dbkey, attr)
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
    dbkey: int
    files = clientprop('poll.files')
    reviews = clientprop('poll.reviews')

class Review(ClientMixin, ReviewSpec):
    dbkey: int
    poll = clientprop('review.poll')
    file = clientprop('review.file')

class File(ClientMixin, FileSpec):
    dbkey: int
    poll = clientprop('file.poll')
    reviews = clientprop('file.reviews')

class PymolCMD(ClientMixin, PymolCMDSpec):
    dbkey: int

class PPPClient:
    def __init__(self, server_addr_or_testclient):
        if isinstance(server_addr_or_testclient, str):
            self.testclient, self.server_addr = None, server_addr_or_testclient
        else:
            self.testclient = server_addr_or_testclient
        assert self.get('/')['msg'] == 'Hello World'

    def getattr(self, thing, dbkey, attr):
        return self.get(f'/getattr/{thing}/{dbkey}/{attr}')

    def get(self, url):
        if self.testclient: response = self.testclient.get(url)
        else: response = requests.get(f'http://{self.server_addr}/ppp{url}')
        if response.status_code != 200: raise PPPClientError(f'GET failed {url} \n {response}')
        return response.json()

    def post(self, url, thing):
        json = thing.json()
        if self.testclient: response = self.testclient.post(url, content=json)
        else: response = requests.post(f'http://{self.server_addr}/ppp{url}', json)
        if response.status_code != 200: raise PPPClientError(f'POST failed {url} {json} \n {response}')
        return response.json()

    def upload(self, thing):
        kind = type(thing).__name__.replace('Spec', '').lower()
        if kind == 'pymolcmd':
            thing = thing.check_cmds()
        if thing._errors: return thing._errors
        return self.post(f'/create/{kind}', thing)

    def pollinfo(self):
        return requests.get(f'http://{self.server_addr}/pollinfo').json()

    def polls(self):
        # return requests.get(f'http://{self.server_addr}/ppp/polls')
        return [Poll(self, **p) for p in self.get('/polls')]

    def reviews(self):
        return [Review(self, **_) for _ in self.get('/reviews')]

    def files(self):
        return [File(self, **_) for _ in self.get('/files')]

    def pymolcmds(self):
        return [PymolCMD(self, **_) for _ in self.get('/pymolcmds')]

    def pymolcmdsdict(self):
        return self.get('/pymolcmds')

    def poll(self, dbkey):
        return Poll(self, **self.get(f'/poll{dbkey}'))

    def poll_fids(self, dbkey):
        return self.get(f'/poll{dbkey}/fids')

    # def create_poll(self, poll):
    # self.post('/poll', json=json.loads(poll.json()))

    def reviews_for_fname(self, fname):
        fname = fname.replace('/', '__DIRSEP__')
        rev = self.get(f'/reviews/byfname/{fname}')
        return [Review(self, **_) for _ in rev]

    def _add_some_cmds(self):
        self.upload(
            PymolCMDSpec(
                name='sym: Make {sym.upper()}',
                cmdstart='from wills_pymol_crap import symgen',
                cmdon=
                f'symgen.make{sym}("$subject", name="sym"); delete $subject; cmd.set_name("sym", "$subject")',
                cmdoff='remove not chain A',
            ))
