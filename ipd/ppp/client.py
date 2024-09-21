import os
from datetime import datetime
import functools
import json
import tempfile
import ipd
import pathlib
from typing import Any, Optional

pydantic = ipd.lazyimport('pydantic', pip=True)
rich = ipd.lazyimport('rich', 'Rich', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
from ordered_set import OrderedSet as ordset
from rich import print

pymol = ipd.lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger')

DTFMT = "%Y-%m-%dT%H:%M:%S.%f"

class PPPClientError(Exception):
    pass

class PollSpec(pydantic.BaseModel):
    name: str = ''
    desc: str = ''
    path: str
    user: str = 'anonymouscoward'
    public: bool = False
    telemetry: bool = False
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    enddate: datetime = pydantic.Field(default_factory=datetime.now)
    props: list[str] = []
    attrs: dict[str, str | int] = {}

    @pydantic.model_validator(mode='after')
    def _update(self):
        if not self.name: self.name = os.path.basename(self.path)
        self.name = self.name.title()
        if not self.desc: self.desc = f"PDBs in {self.path}"
        return self

    @pydantic.validator('props')
    def _valprops(cls, v):
        return list(v)

    def __hash__(self):
        return hash(self.path)

class ReviewSpec(pydantic.BaseModel):
    polldbkey: int
    fname: str
    grade: str
    user: str = 'anonymouscoward'
    comment: str = ''
    durationsec: int = -1
    props: list[str] = []
    attrs: dict[str, str | int | float] = {}
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)

    @pydantic.validator('grade')
    def valgrade(cls, grade):
        assert grade.upper() in 'SABCDF'
        return grade.upper()

    @pydantic.validator('fname')
    def valfname(cls, fname):
        assert os.path.exists(fname)
        return os.path.abspath(fname)

class FileSpec(pydantic.BaseModel):
    polldbkey: int
    fname: str
    public: bool = True
    user: str = 'anonymouscoward'
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    props: list[str] = []
    attrs: dict[str, str | int | float] = {}

    @pydantic.validator('fname')
    def valfname(cls, fname):
        assert os.path.exists(fname)
        return os.path.abspath(fname)

class PymolCMDSpecError(Exception):
    def __init__(self, message, log):
        super().__init__(message + os.linesep+ log)
        self.log = log

class PymolCMDSpec(pydantic.BaseModel):
    name: str
    desc: str = ''
    cmdon: str
    cmdoff: str
    cmdstart: str = ''
    onstart: bool = False
    public: bool = True
    user: str = 'anonymouscoward'
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    props: list[str] = []
    attrs: dict[str, str | int | float] = {}

    @pydantic.model_validator(mode='after')
    def check_cmds(self):
        self._check_cmds_output = '-'*80 + os.linesep
        self.check_cmd('cmdstart')
        self.check_cmd('cmdon')
        self.check_cmd('cmdoff')
        if any(self._check_cmds_output.lower().count(err) for err in 'error unknown unrecognized'.split()):
            raise PymolCMDSpecError('bad pymol commands', self._check_cmds_output)
        return self

    def check_cmd(self, cmdname):
        with tempfile.TemporaryDirectory() as td:
            with open(f'{td}/stdout.log', 'w') as out:
                with ipd.dev.redirect(stdout=out, stderr=out):
                    pymol.cmd.do(getattr(self, cmdname), echo=False)
            msg = pathlib.Path(f'{td}/stdout.log').read_text()
            msg = cmdname.upper() + os.linesep + msg + '-' * 80 + os.linesep
            self._check_cmds_output += msg
        return self

class ClientMixin(pydantic.BaseModel):
    _pppclient: Optional['PPPClient'] = None

    def __init__(self, client, **kw):
        super().__init__(**kw)
        self._pppclient = client

    def __hash__(self):
        return self.dbkey

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
    pymoldbkey: int

class PPPClient:
    def __init__(self, server_addr_or_testclient):
        if isinstance(server_addr_or_testclient, str): self.server_addr = server_addr_or_testclient
        else: self.testclient = server_addr_or_testclient
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
        else: requests.post(f'http://{self.server_addr}/ppp{url}', content=json)
        if response.status_code != 200: raise PPPClientError(f'POST failed {url} {json} \n {response}')
        return response.json()

    def upload(self, thing):
        kind = type(thing).__name__.replace('Spec', '').lower()
        return self.post(f'/create/{kind}', thing)

        # if isinstance(thing, PollSpec): return self.create_poll(thing)
        # if isinstance(thing, ReviewSpec): return self.create_review(thing)
        # if isinstance(thing, PymolCMDSpec): return self.create_pymolcmd(thing)
        # else: raise ValueError(f'dont know how to post {type(thing)}')

    def polls(self):
        return [Poll(self, **p) for p in self.get('/polls')]

    def reviews(self):
        return [Review(self, **_) for _ in self.get('/reviews')]

    def files(self):
        return [File(self, **_) for _ in self.get('/files')]

    def pymolcmds(self):
        return [PymolCMD(self, **_) for _ in self.get('/pymolcmds')]

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
