from datetime import datetime
from typing import Union
import getpass
import socket
import os
from ipd.dev.lazy_import import lazyimport

pydantic = lazyimport('pydantic', pip=True)
STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
SERVER = 'ppp' == socket.gethostname()

def set_server(isserver):
    global SERVER
    SERVER = isserver
    print('SERVER MODE')

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
    props: Union[list[str], str] = []
    attrs: Union[dict[str, Union[str, int, float]], str] = {}
    _errors: str = ''

    @pydantic.validator('user')
    def valuser(cls, user):
        assert user != 'services'
        return user or getpass.getuser()

    @pydantic.validator('props')
    def valprops(cls, props):
        if isinstance(props, (set, list)): return props
        try:
            props = ipd.dev.safe_eval(props)
        except (NameError, SyntaxError):
            if isinstance(props, str):
                if not props.strip(): return []
                props = [p.strip() for p in props.strip().split(',')]
        return props

    @pydantic.validator('attrs')
    def valattrs(cls, attrs):
        if isinstance(attrs, dict): return attrs
        try:
            ipd.dev.safe_eval(attrs)
        except (NameError, SyntaxError):
            if isinstance(attrs, str):
                if not attrs.strip(): return {}
                attrs = {
                    x.split('=').split(':')[0].strip(): x.split('=').split(':')[1].strip()
                    for x in attrs.strip().split(',')
                }
        return attrs

    def errors(self):
        return self._errors

    def __getitem__(self, k):
        return getattr(self, k)

_checkobjnum = 0

class PollSpec(SpecBase):
    name: str
    desc: str = ''
    path: str
    cmdstart: str = ''
    cmdstop: str = ''
    sym: str = ''
    nchain: Union[int, str] = -1
    ligand: str = ''
    workflowname: str = ''
    enddate: datetime = datetime.strptime('9999-01-01T01:01:01.1', DATETIME_FORMAT)
    _temporary = False

    @pydantic.validator('nchain')
    def valnchain(cls, nchain):
        return int(nchain)

    @pydantic.validator('path')
    def valpath(cls, path):
        print('valpath', path)
        if SERVER: return path
        if digs := path.startswith('digs:'): path = path[5:]
        path = os.path.abspath(os.path.expanduser(path))
        if digs or not os.path.exists(path):
            assert check_output(['rsync', f'digs:{path}']), f'path {path} must exist locally or on digs'
        else:
            assert os.path.isdir(path), f'path must be directory: {path}'
        return f'digs:{path}' if digs else path

    @pydantic.model_validator(mode='after')
    def _validated(self):
        # sourcery skip: merge-duplicate-blocks, remove-redundant-if, set-comprehension, split-or-ifs
        print('poll _validated')
        if SERVER: return self
        print('poll _validated not server')
        fix_label_case(self)
        if self.path.startswith('digs:'): return self
        self.name = self.name or os.path.basename(self.path)
        self.desc = self.desc or f'PDBs in {self.path}'
        self.sym = self.sym or guess_sym_from_directory(self.path, suffix=STRUCTURE_FILE_SUFFIX)
        if not self.sym or not self.ligand:
            try:
                global _checkobjnum
                filt = lambda s: not s.startswith('_') and s.endswith(STRUCTURE_FILE_SUFFIX)
                fname = next(filter(filt, os.listdir(self.path)))
                # print('CHECKING IN PYMOL', fname)
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
                    self.sym = guess_symmetry(xyz)
            except ValueError:
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

class ReviewSpec(SpecBase):
    polldbkey: int
    fname: str
    permafname: str = ''
    grade: str
    comment: str = ''
    flowstepname: str = ''
    durationsec: int = -1
    filecontent: Union[str, None] = None

    @pydantic.validator('grade')
    def valgrade(cls, grade):
        assert grade in 'like superlike dislike hate'.split()
        return grade

    @pydantic.validator('fname')
    def valfname(cls, fname):
        if SERVER: return fname
        assert os.path.exists(fname)
        return os.path.abspath(fname)

    @pydantic.model_validator(mode='after')
    def _validated(self):
        fix_label_case(self)
        if self.grade == 'S' and not self.comment:
            self._errors += 'S-tier review requires a comment!'
        return self

class FileSpec(SpecBase):
    polldbkey: int
    fname: str
    permafname: str = ''
    filecontent: str = ''

    @pydantic.validator('fname')
    def valfname(cls, fname):
        if SERVER or REMOTE_MODE: return fname
        fname = os.path.abspath(fname)
        assert os.path.exists(fname)  # or check_output(['rsync', f'digs:{fname}'])
        return fname

class PymolCMDSpecError(Exception):
    def __init__(self, message, log):
        super().__init__(message + os.linesep + log)
        self.log = log

TOBJNUM = 0

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
    maxchains: int = 9999
    cmdcheck: bool = True

    @pydantic.model_validator(mode='after')
    def _validated(self):
        fix_label_case(self)
        if self.cmdcheck: self._check_cmds()
        return self

    def _check_cmds(self):
        pymol.cmd.save('/tmp/tmp_pymol_session.pse')
        self._check_cmds_output = '-' * 80 + os.linesep + str(self) + os.linesep + '_' * 80 + os.linesep
        self._errors = ''
        global TOBJNUM
        TOBJNUM += 1
        pymol.cmd.load(ipd.testpath('pdb/tiny.pdb'), f'TEST_OBJECT{TOBJNUM}')
        self._check_cmd('cmdstart')
        self._check_cmd('cmdon')
        self._check_cmd('cmdoff')
        pymol.cmd.delete(f'TEST_OBJECT{TOBJNUM}')
        pymol.cmd.load('/tmp/tmp_pymol_session.pse')
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
            pymol.cmd.do('delete *TMP*')
            msg = Path(f'{td}/stdout.log').read_text()
            msg = cmdname.upper() + os.linesep + msg + '-' * 80 + os.linesep
            self._check_cmds_output += msg
        return self

class FlowStepSpec(SpecBase):
    name: str
    index: int
    taskgen: str
    cmdnames: list[str]
    workflowname: str
    instructions: ''

class WorkflowSpec(SpecBase):
    name: str
    desc: str = ''
    ordering: str
    steps: list[FlowStepSpec]

    @pydantic.field_validator('ordering')
    def check_ordering(ordering):
        allowed = ['parallel', 'parallel-steps', 'sequential', 'gridview']
        assert ordering in allowed, f'bad ordering {ordering}, must be one of {allowed}'
        return ordering
