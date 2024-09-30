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
# profile = ipd.dev.timed
profile = lambda f: f

class PPPClientError(Exception):
    pass

_checkobjnum = 0

def fix_label_case(thing):
    if isinstance(thing, dict):
        keys, get, set = thing.keys(), thing.__getitem__, thing.__setitem__
    elif isinstance(thing, pydantic.BaseModel):
        keys, get, set = thing.model_fields_set, thing.__getattribute__, thing.__setattr__
    else:
        raise TypeError(f' fix_label_case unsupported type{type(thing)}')
    if 'name' in keys: set('name', get('name').title())
    if 'sym' in keys: set('sym', get('sym').upper())
    if 'user' in keys: set('user', get('user').lower())
    if 'workflow' in keys: set('workflow', get('workflow').lower())
    if 'ligand' in keys: set('ligand', get('ligand').upper())

class SpecBase(pydantic.BaseModel):
    ispublic: bool = True
    telemetry: bool = True
    user: str = ''
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    props: list[str] | str = []
    attrs: dict[str, str | int | float] | str = {}
    _errors: str = ''

    @pydantic.validator('user')
    def valuser(cls, user):
        return user or getpass.getuser()

    @pydantic.validator('props')
    def valprops(cls, props):
        if isinstance(props, str):
            props = [p.strip() for p in props.split(',')]
        return props

    @pydantic.validator('attrs')
    def valattrs(cls, attrs):
        if isinstance(attrs, str):
            attrs = {x.split('=')[0].strip(): x.split('=')[1].strip() for x in attrs.split(',')}
        return attrs

    def errors(self):
        return self._errors

    def __getitem__(self, k):
        return getattr(self, k)

@profile
class PollSpec(SpecBase):
    name: str
    desc: str = ''
    path: str
    cmdstart: str = ''
    cmdstop: str = ''
    sym: str = ''
    nchain: int | str = -1
    ligand: str = ''
    workflow: str = ''
    enddate: datetime = datetime.strptime('9999-01-01T01:01:01.1', DATETIME_FORMAT)
    _temporary = False

    @pydantic.validator('nchain')
    def valnchain(cls, nchain):
        return int(nchain)

    @pydantic.validator('path')
    def valpath(cls, path):
        path = os.path.abspath(os.path.expanduser(path))
        assert os.path.isdir(path), f'path must be directory: {path}'
        assert [f for f in os.listdir(path)
                if f.endswith(STRUCTURE_FILE_SUFFIX)], f'path must contain structure files: {path}'
        return path

    @pydantic.model_validator(mode='after')
    def _validated(self):
        # sourcery skip: merge-duplicate-blocks, remove-redundant-if, set-comprehension, split-or-ifs
        fix_label_case(self)
        if not self.name: self.name = os.path.basename(self.path)
        if not self.desc: self.desc = f'PDBs in {self.path}'
        self.sym = ipd.sym.guess_sym_from_directory(self.path, suffix=STRUCTURE_FILE_SUFFIX)
        if not self.sym or self.ligand == '':
            try:
                global _checkobjnum
                filt = lambda s: not s.startswith('_') and s.endswith(STRUCTURE_FILE_SUFFIX)
                fname = next(filter(filt, os.listdir(self.path)))
                pymol.cmd.set('suspend_updates', 'on')
                with contextlib.suppress(pymol.CmdException):
                    pymol.cmd.save(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
                pymol.cmd.delete('all')
                pymol.cmd.load(os.path.join(self.path, fname), f'TMP{_checkobjnum}')
                if self.ligand == '':
                    ligs = set()
                    for a in pymol.cmd.get_model(f'TMP{_checkobjnum} and HET and not resn HOH').atom:
                        ligs.add(a.resn)
                    self.ligand = ','.join(ligs)
                pymol.cmd.remove(f'TMP{_checkobjnum} and not name ca')
                chains = {a.chain for a in pymol.cmd.get_model(f'TMP{_checkobjnum}').atom}
                self.nchain = len(chains)
                if not self.sym:
                    xyz = pymol.cmd.get_coords(f'TMP{_checkobjnum}')
                    if xyz is None:
                        self._errors += f'pymol get_coords failed on TMP{_checkobjnum}\nfname: {fname}'
                        return self
                    xyz = xyz.reshape(len(chains), -1, 3)
                    self.sym = ipd.sym.guess_symmetry(xyz)
            except ValueError as e:
                # print(os.path.join(self.path, fname))
                traceback.print_exc()
                if self.nchain < 4: self.sym = 'C1'
                else: self.sym = 'unknown'
            except (AttributeError, pymol.CmdException, gzip.BadGzipFile) as e:
                self._errors += f'POLL error in _validated: {type(e)}\n{e}'
            finally:
                pymol.cmd.delete(f'TMP{_checkobjnum}')
                pymol.cmd.set('suspend_updates', 'off')
                _checkobjnum += 1
                with contextlib.suppress(pymol.CmdException):
                    pymol.cmd.load(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
                    os.remove(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
        assert isinstance(self.nchain, int)
        return self

    @pydantic.validator('props')
    def _valprops(cls, v):
        return list(v)

    def __hash__(self):
        return hash(self.path)

@profile
class ReviewSpec(SpecBase):
    polldbkey: int
    fname: str
    permafile: str = ''
    grade: str
    comment: str = ''
    durationsec: int = -1

    @pydantic.validator('grade')
    def valgrade(cls, grade, values):
        assert grade.upper() in 'SABCDF'
        return grade.upper()

    @pydantic.validator('fname')
    def valfname(cls, fname):
        assert os.path.exists(fname)
        return os.path.abspath(fname)

    @pydantic.model_validator(mode='after')
    def _validated(self):
        fix_label_case(self)
        if self.grade == 'S' and not self.comment:
            self._errors += 'S-tier review requires a comment!'
        return self

@profile
class FileSpec(SpecBase):
    polldbkey: int
    fname: str

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
class PymolCMDSpec(SpecBase):
    name: str
    desc: str = ''
    cmdon: str
    cmdoff: str = ''
    cmdstart: str = ''
    onstart: bool = False
    ligand: str = ''
    sym: str = ''
    minchains: int = 1
    maxchains: int = 999_999_999

    @pydantic.model_validator(mode='after')
    def _validated(self):
        fix_label_case(self)
        self._check_cmds()
        return self

    def _check_cmds(self):
        self._check_cmds_output = '-' * 80 + os.linesep + str(self) + os.linesep + '_' * 80 + os.linesep
        self._errors = ''
        global TOBJNUM
        TOBJNUM += 1
        pymol.cmd.load(ipd.testpath('pdb/tiny.pdb'), f'TEST_OBJECT{TOBJNUM}')
        self._check_cmd('cmdstart')
        self._check_cmd('cmdon')
        self._check_cmd('cmdoff')
        pymol.cmd.delete(f'TEST_OBJECT{TOBJNUM}')
        if any(
            [any(self._check_cmds_output.lower().count(err) for err in 'error unknown unrecognized'.split())]):
            # raise PymolCMDSpecError('bad pymol commands', self._check_cmds_output)
            self._errors = self._check_cmds_output
        return self

    def _check_cmd(self, cmdname):
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

    def get(self, url, **kw):
        fix_label_case(kw)
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        if not self.testclient: url = f'http://{self.server_addr}/ppp{url}'
        if self.testclient: response = self.testclient.get(url)
        else: response = requests.get(url)
        if response.status_code != 200:
            raise PPPClientError(f'GET failed URL: "{url}" \n RESPONSE: "{response.reason}"')
        return response.json()

    def post(self, url, thing, **kw):
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        if not self.testclient: url = f'http://{self.server_addr}/ppp{url}'
        # print('POST', url, thing)
        json = thing.json()
        if self.testclient: response = self.testclient.post(url, content=json)
        else: response = requests.post(url, json)
        if response.status_code != 200: raise PPPClientError(f'POST failed {url} {json} \n {response}')
        return response.json()

    def upload(self, thing, **kw):
        # print('upload', type(thing), kw)
        if thing._errors: return thing._errors
        kind = type(thing).__name__.replace('Spec', '').lower()
        return self.post(f'/create/{kind}', thing, **kw)

    def pollinfo(self, user=getpass.getuser()):
        if self.testclient: return self.testclient.get(f'/pollinfo?user={user}').json()
        print(requests.get(f'http://{self.server_addr}/pollinfo?user={user}'))
        return requests.get(f'http://{self.server_addr}/pollinfo?user={user}').json()

    def polls(self, **kw):
        return [Poll(self, **p) for p in self.get('/polls', **kw)]

    def reviews(self, **kw):
        return [Review(self, **_) for _ in self.get('/reviews', **kw)]

    def files(self, **kw):
        return [File(self, **_) for _ in self.get('/files', **kw)]

    def pymolcmds(self, **kw):
        return [PymolCMD(self, **_) for _ in self.get('/pymolcmds', **kw)]

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
