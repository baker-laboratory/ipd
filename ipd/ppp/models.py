import contextlib
from datetime import datetime
import os
import tempfile
from pathlib import Path
from subprocess import check_output
from typing import Union

import pydantic

import ipd
import ipd.ppp
from ipd.crud import ModelRef, Unique
from ipd.sym.guess_symmetry import guess_sym_from_directory, guess_symmetry

class SpecWithProps(ipd.crud.SpecBase):
    props: Union[list[str], str] = []
    attrs: Union[dict[str, Union[str, int, float]], str] = {}

    @pydantic.field_validator('props')
    def valprops(cls, props):
        if isinstance(props, (set, list)): return props
        try:
            props = ipd.dev.safe_eval(props)
        except (NameError, SyntaxError):
            if isinstance(props, str):
                if not props.strip(): return []
                props = [p.strip() for p in props.strip().split(',')]

    @pydantic.field_validator('attrs')
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

class SpecWithUser(ipd.crud.SpecBase):
    userid: ModelRef['UserSpec'] = pydantic.Field(default='anonymous_coward', validate_default=True)
    ispublic: bool = True
    telemetry: bool = False

class PollSpec(SpecWithUser, SpecWithProps):
    name: Unique[str]
    desc: str = ''
    path: str
    cmdstart: str = ''
    cmdstop: str = ''
    sym: str = ''
    nchain: int = -1
    ligand: str = ''
    workflowid: ModelRef['WorkflowSpec'] = None
    enddate: datetime = datetime.strptime('9999-01-01T01:01:01.1', ipd.DATETIME_FORMAT)

    @pydantic.field_validator('nchain')
    def valnchain(cls, nchain):
        return int(nchain)

    def valpath(self, path):
        if self.ghost or ipd.ppp.server.servermode(): return path
        if digs := path.startswith('digs:'): path = path[5:]
        path = os.path.abspath(os.path.expanduser(path))
        if digs or not os.path.exists(path):
            assert check_output(['rsync', f'digs:{path}']), f'path {path} must exist locally or on digs'
            print('TODO: check for struct files')
        else:
            assert os.path.isdir(path), f'path must be directory: {path}'
            files = list(filter(lambda s: s.endswith(ipd.STRUCTURE_FILE_SUFFIX), os.listdir(path)))
            assert files, f'no files in {path} end with {ipd.STRUCTURE_FILE_SUFFIX}'
        return f'digs:{path}' if digs else path

    @pydantic.model_validator(mode='after')
    def _validated(self):
        # sourcery skip: merge-duplicate-blocks, remove-redundant-if, set-comprehension, split-or-ifs
        # if ipd.ppp.server.servermode(): return self  # client does validation
        # print('poll _validated not server')
        if self.id is not None: return self
        fix_label_case(self)
        if self.path.startswith('digs:'): return self
        self.path = self.valpath(self.path)
        self.name = self.name or os.path.basename(self.path)
        self.desc = self.desc or f'PDBs in {self.path}'
        self.sym = self.sym or guess_sym_from_directory(self.path, suffix=ipd.STRUCTURE_FILE_SUFFIX)
        self = PollSpec_get_structure_properties(self)
        # print('poll _validated done')
        return self

    @ipd.crud.backend_method
    def clear(self, backend, ghost=True):
        for r in backend.select(backend.DBReview, pollid=self.id):
            if not ghost: r.pollid, r.pollfileid = 1, 1
        backend.session.commit()
        for f in backend.select(backend.DBPollFile, pollid=self.id):
            if ghost: f.ghost = True
            else: backend.session.delete(f)

    def __hash__(self):
        return hash(self.path)

class FileKindSpec(ipd.crud.SpecBase):
    filekind: str

class PollFileSpec(ipd.crud.SpecBase):
    pollid: ModelRef['PollSpec']
    fname: str
    tag: str = ''
    permafname: str = ''
    filecontent: str = ''
    parentid: ModelRef['PollFileSpec', 'children'] = None
    filekindid: ModelRef['FileKindSpec'] = None

    @pydantic.field_validator('fname')
    def valfname(cls, fname):
        if ipd.ppp.server.servermode() or ipd.ppp.REMOTE_MODE: return fname
        fname = os.path.abspath(fname)
        assert os.path.exists(fname)  # or check_output(['rsync', f'digs:{fname}'])
        return fname

class ReviewSpec(SpecWithUser):
    pollid: ModelRef['PollSpec']
    grade: str
    comment: str = ''
    pollfileid: ModelRef['PollFileSpec']
    workflowid: ModelRef['WorkflowSpec'] = pydantic.Field(default='Manual', validate_default=True)
    durationsec: int = -1

    @pydantic.field_validator('grade')
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

class ReviewStepSpec(ipd.crud.SpecBase):
    reviewid: ModelRef['ReviewSpec']
    flowstepid: ModelRef['FlowStepSpec']
    task: dict[str, Union[str, int, float]]
    grade: str
    comment: str = ''
    durationsec: int = -1

class PymolCMDSpecError(Exception):
    def __init__(self, message, log):
        super().__init__(message + os.linesep + log)
        self.log = log

class PymolCMDSpec(SpecWithUser):
    name: Unique[str]
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

class WorkflowSpec(SpecWithUser):
    name: Unique[str]
    desc: str
    ordering: str = 'Manual'

    @pydantic.field_validator('ordering')
    def check_ordering(ordering):
        allowed = ['Manual', 'Parallel', 'Parallel Steps', 'Sequential', 'Grid View']
        assert ordering in allowed, f'bad ordering {ordering}, must be one of {allowed}'
        return ordering

class FlowStepSpec(ipd.crud.SpecBase):
    workflowid: ModelRef['WorkflowSpec']
    name: Unique[str]
    index: int
    taskgen: dict[str, Union[str, int, float]] = {}
    instructions: str = ''

class UserSpec(ipd.crud.SpecBase):
    name: Unique[str]
    fullname: str = ''
    followers: list['UserSpec'] = []
    following: list['UserSpec'] = []
    groups: list['GroupSpec'] = []

class GroupSpec(SpecWithUser):
    name: Unique[str]
    users: list['UserSpec'] = []
    userid: ModelRef['UserSpec', 'ownedgroups'] = pydantic.Field(default='anonymous_coward',
                                                                 validate_default=True)

_PML = 0

def PollSpec_get_structure_properties(poll):
    if not poll.sym or not poll.ligand:
        try:
            global _PML
            filt = lambda s: not s.startswith('_') and s.endswith(ipd.STRUCTURE_FILE_SUFFIX)
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
            poll.sym = 'C1' if poll.nchain < 4 else 'unknown'
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
    import pymol
    pymol.cmd.save('/tmp/tmp_pymol_session.pse')
    command._check_cmds_output = '-' * 80 + os.linesep + str(command) + os.linesep + '_' * 80 + os.linesep
    command._errors = ''
    global _PML
    _PML += 1
    pymol.cmd.load(ipd.dev.package_testdata_path('pdb/tiny.pdb'), f'TEST_OBJECT{_PML}')
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
    import pymol
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

def fix_label_case(thing):
    if isinstance(thing, dict):
        keys, get, set = thing.keys(), thing.__getitem__, thing.__setitem__
    elif isinstance(thing, pydantic.BaseModel):
        keys, get, set = thing.model_fields_set, thing.__getattribute__, thing.__setattr__
    else:
        raise TypeError(f' fix_label_case unsupported type{type(thing)}')
    if 'sym' in keys: set('sym', get('sym').upper())
    if 'ligand' in keys: set('ligand', get('ligand').upper())
    return thing

spec_models = {
    name.replace('Spec', '').lower(): cls
    for name, cls in globals().items() if name.endswith('Spec')
}
assert not any(name.endswith('s') for name in spec_models)
