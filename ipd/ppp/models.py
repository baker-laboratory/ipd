from datetime import datetime
from typing import Union
import getpass
import socket
import contextlib
import os
import ipd
import tempfile
from pathlib import Path
from ipd.dev.lazy_import import lazyimport
from ipd.sym.guess_symmetry import guess_symmetry, guess_sym_from_directory

pydantic = lazyimport('pydantic', pip=True)
pymol = lazyimport('pymol')

STRUCTURE_FILE_SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
_SERVERMODE = 'ppp' == socket.gethostname()

def set_servermode(isserver):
    global _SERVERMODE
    _SERVERMODE = isserver
    # print('_SERVERMODE MODE')

def fix_label_case(thing):
    if isinstance(thing, dict):
        keys, get, set = thing.keys(), thing.__getitem__, thing.__setitem__
    elif isinstance(thing, pydantic.BaseModel):
        keys, get, set = thing.model_fields_set, thing.__getattribute__, thing.__setattr__
    else:
        raise TypeError(f' fix_label_case unsupported type{type(thing)}')
    if 'sym' in keys: set('sym', get('sym').upper())
    if 'ligand' in keys: set('ligand', get('ligand').upper())

class SpecBase(pydantic.BaseModel):
    ispublic: bool = True
    telemetry: bool = False
    ghost: bool = False
    datecreated: datetime = pydantic.Field(default_factory=datetime.now)
    props: Union[list[str], str] = []
    attrs: Union[dict[str, Union[str, int, float]], str] = {}
    _errors: str = ''

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
            attrs = ipd.dev.safe_eval(attrs)
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

    def spec(self):
        return self

class StrictFields:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)

        def new_init(self, pppclient=None, id=None, **data):
            for name in data:
                if name not in cls.__fields__:
                    raise TypeError(f"{cls} Invalid field name: {name}")
            data |= dict(pppclient=pppclient, id=id)
            super(cls, self).__init__(**data)

        cls.__init__ = new_init

class _SpecWithUser(SpecBase):
    userid: Union[str, int] = pydantic.Field(default='anonymous_coward', validate_default=True)

    @pydantic.validator('userid')
    def valuserid(userid):
        # print('VALUSER', userid)
        if isinstance(userid, str):
            client = ipd.ppp.get_hack_fixme_global_client()
            assert client, 'client unavailable'
            if user := client.user(name=userid):
                # print('FOUNDUSER', user.name, user.id)
                return user.id
            # print('MISSING USER', userid)
            raise ValueError(f'unknown user "{userid}"')
        return int(userid)

    @pydantic.model_validator(mode='after')
    def validate(self):
        assert isinstance(self.userid, int), 'userid must be an int after validation'
        return self

class PollSpec(_SpecWithUser, StrictFields):
    name: str
    desc: str = ''
    path: str
    cmdstart: str = ''
    cmdstop: str = ''
    sym: str = ''
    nchain: Union[int, str] = -1
    ligand: str = ''
    workflowid: int = 1
    enddate: datetime = datetime.strptime('9999-01-01T01:01:01.1', DATETIME_FORMAT)

    @pydantic.validator('nchain')
    def valnchain(cls, nchain):
        return int(nchain)

    @pydantic.validator('path')
    def valpath(cls, path):
        # print('valpath', path)
        if _SERVERMODE: return path
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
        if _SERVERMODE: return self  # client does validation
        print('poll _validated not server')
        fix_label_case(self)
        if self.path.startswith('digs:'): return self
        self.name = self.name or os.path.basename(self.path)
        self.desc = self.desc or f'PDBs in {self.path}'
        self.sym = self.sym or guess_sym_from_directory(self.path, suffix=STRUCTURE_FILE_SUFFIX)
        self = PollSpec_get_structure_properties(self)
        print('poll _validated done')
        return self

    def __hash__(self):
        return hash(self.path)

class FileKindSpec(SpecBase, StrictFields):
    kind: str

class PollFileSpec(SpecBase, StrictFields):
    pollid: int
    fname: str
    tag: str = ''
    permafname: str = ''
    filecontent: str = ''
    parentid: Union[int, None] = None
    filekindid: Union[int, None] = None

    @pydantic.validator('fname')
    def valfname(cls, fname):
        if _SERVERMODE or REMOTE_MODE: return fname
        fname = os.path.abspath(fname)
        assert os.path.exists(fname)  # or check_output(['rsync', f'digs:{fname}'])
        return fname

class ReviewSpec(_SpecWithUser, StrictFields):
    pollid: int
    grade: str
    comment: str = ''
    pollfileid: Union[str, int]
    workflowid: Union[str, int] = pydantic.Field(default='Manual', validate_default=True)
    durationsec: int = -1

    @pydantic.validator('workflowid')
    def valflowid(workflowid):
        if isinstance(workflowid, str):
            client = ipd.ppp.get_hack_fixme_global_client()
            if not client: return workflowid
            if workflow := client.workflow(name=workflowid): return workflow.id
            raise ValueError(f'unknown workflow "{workflowid}"')
        return int(workflowid)

    @pydantic.validator('grade')
    def valgrade(cls, grade):
        vals = 'like superlike dislike hate'.split()
        assert grade in vals, f'grade {grade} not in allowed: {vals}'
        return grade

    @pydantic.model_validator(mode='after')
    def _validated(self):
        if isinstance(self.pollfileid, str):
            client = ipd.ppp.get_hack_fixme_global_client()
            if pfile := client.pollfile(pollid=self.pollid, fname=self.pollfileid):
                self.pollfileid = pfile.id
            else:
                raise ValueError(f'unknown poll file "{self.pollfileid}" in poll {self.pollid}')
        if self.grade == 'superlike' and not self.comment:
            self._errors += 'Super-Like requires a comment!'
        return self

class ReviewStepSpec(SpecBase, StrictFields):
    reviewid: int
    flowstepid: int
    task: dict[str, Union[str, int, float]]
    grade: str
    comment: str = ''
    durationsec: int = -1

class PymolCMDSpecError(Exception):
    def __init__(self, message, log):
        super().__init__(message + os.linesep + log)
        self.log = log

class PymolCMDSpec(_SpecWithUser, StrictFields):
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
        if self.cmdcheck: PymolCMDSpec_validate_commands(self)
        return self

class WorkflowSpec(_SpecWithUser, StrictFields):
    name: str
    desc: str
    ordering: str = 'Manual'

    @pydantic.field_validator('ordering')
    def check_ordering(ordering):
        allowed = ['Manual', 'Parallel', 'Parallel Steps', 'Sequential', 'Grid View']
        assert ordering in allowed, f'bad ordering {ordering}, must be one of {allowed}'
        return ordering

class FlowStepSpec(SpecBase, StrictFields):
    workflowid: int
    name: str
    index: int
    taskgen: dict[str, Union[str, int, float]] = {}
    instructions: str = ''

class UserSpec(SpecBase):
    name: str
    fullname: str = ''

class GroupSpec(_SpecWithUser, StrictFields):
    name: str

_PML = 0

def PollSpec_get_structure_properties(poll):
    if not poll.sym or not poll.ligand:
        try:
            global _PML
            filt = lambda s: not s.startswith('_') and s.endswith(STRUCTURE_FILE_SUFFIX)
            fname = next(filter(filt, os.listdir(poll.path)))
            # print('CHECKING IN PYMOL', fname)
            pymol.cmd.set('suspend_updates', 'on')
            with contextlib.suppress(pymol.CmdException):
                pymol.cmd.save(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
            pymol.cmd.delete('all')
            pymol.cmd.load(os.path.join(poll.path, fname), f'TMP{_PML}')
            if poll.ligand == '':
                ligs = {a.resn for a in pymol.cmd.get_model(f'TMP{_PML} and HET and not resn HOH').atom}
                poll.ligand = ','.join(ligs)
            pymol.cmd.remove(f'TMP{_PML} and not name ca')
            chains = {a.chain for a in pymol.cmd.get_model(f'TMP{_PML}').atom}
            poll.nchain = len(chains)
            if not poll.sym:
                xyz = pymol.cmd.get_coords(f'TMP{_PML}')
                if xyz is None:
                    poll._errors += f'pymol get_coords failed on TMP{_PML}\nfname: {fname}'
                    return poll
                xyz = xyz.reshape(len(chains), -1, 3)
                poll.sym = guess_symmetry(xyz)
        except ValueError:
            # print(os.path.join(poll.path, fname))
            traceback.print_exc()
            if poll.nchain < 4: poll.sym = 'C1'
            else: poll.sym = 'unknown'
        except (AttributeError, pymol.CmdException, gzip.BadGzipPollFile) as e:
            poll._errors += f'POLL error in _validated: {type(e)}\n{e}'
        finally:
            pymol.cmd.delete(f'TMP{_PML}')
            pymol.cmd.set('suspend_updates', 'off')
            _PML += 1
            with contextlib.suppress(pymol.CmdException):
                pymol.cmd.load(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
                os.remove(os.path.expanduser('~/.config/ppp/poll_check_save.pse'))
    assert isinstance(poll.nchain, int)
    return poll

def PymolCMDSpec_validate_commands(command):
    pymol.cmd.save('/tmp/tmp_pymol_session.pse')
    command._check_cmds_output = '-' * 80 + os.linesep + str(command) + os.linesep + '_' * 80 + os.linesep
    command._errors = ''
    global _PML
    _PML += 1
    pymol.cmd.load(ipd.testpath('pdb/tiny.pdb'), f'TEST_OBJECT{_PML}')
    PymolCMDSpec_validate_command(command, 'cmdstart')
    PymolCMDSpec_validate_command(command, 'cmdon')
    PymolCMDSpec_validate_command(command, 'cmdoff')
    pymol.cmd.delete(f'TEST_OBJECT{_PML}')
    pymol.cmd.load('/tmp/tmp_pymol_session.pse')
    if any(
        [any(command._check_cmds_output.lower().count(err) for err in 'error unknown unrecognized'.split())]):
        # raise PymolCMDSpecError('bad pymol commands', command._check_cmds_output)
        command._errors = command._check_cmds_output
    return command

def PymolCMDSpec_validate_command(command, cmdname):
    with tempfile.TemporaryDirectory() as td:
        with open(f'{td}/stdout.log', 'w') as out:
            with ipd.dev.redirect(stdout=out, stderr=out):
                cmd = getattr(command, cmdname)
                cmd = cmd.replace('$subject', f'TEST_OBJECT{_PML}')
                pymol.cmd.do(cmd, echo=False, log=False)
        pymol.cmd.do('delete *TMP*')
        msg = Path(f'{td}/stdout.log').read_text()
        msg = cmdname.upper() + os.linesep + msg + '-' * 80 + os.linesep
        command._check_cmds_output += msg
    return command

spec_model = {
    name.replace('Spec', '').lower(): cls
    for name, cls in globals().items() if name.endswith('Spec')
}
assert not any(name.endswith('s') for name in spec_model)
