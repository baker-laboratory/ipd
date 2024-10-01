from _pickle import PicklingError
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from icecream import ic
import abc
import contextlib
import datetime
import collections
import ipd
from ipd import ppp
from io import StringIO
import os
import pickle
import pymol
import random
import shutil
import getpass
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from rich import print

it = ipd.lazyimport('itertools', 'more_itertools', pip=True)
requests = ipd.lazyimport('requests', pip=True)
fuzzyfinder = ipd.lazyimport('fuzzyfinder', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)

remote, state, ppppp = None, None, None
ISGLOBALSTATE, ISPERPOLLSTATE = set(), set()
SERVER_ADDR = os.environ.get('PPPSERVER', 'jojo:12345')
CONFIG_DIR = os.path.expanduser('~/.config/ppp/')
CONFIG_FILE = f'{CONFIG_DIR}/localconfig.yaml'
STATE_FILE = f'{CONFIG_DIR}/localstate.yaml'
SESSION_RESTORE = f'{CONFIG_DIR}/session_restore.pse'
PPPPP_PICKLE = f'{CONFIG_DIR}/PrettyProteinProjectPymolPluginPanel.pickle'
TEST_STATE = {}
DEFAULTS = dict(reviewed=set(),
                pymol_view={},
                prefetch=True,
                review_action='cp $file $pppdir/$poll/$grade_$filebase',
                do_review_action=False,
                findcmd='',
                findpoll='',
                shuffle=False,
                use_rsync=False,
                hide_invalid=True,
                showallcmds=False,
                pymol_sc_repr='sticks',
                activepoll=None)
# profile = ipd.dev.timed
profile = lambda f: f
_debug_state = {}

def notify(message):
    pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)

def isfalse_notify(ok, message):
    if not ok:
        pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)
        return True

def ppp_pymol_add_default(name, val, isglobal=False):
    name = f'ppp_pymol_{name}'
    DEFAULTS[name] = val
    if state: state._statetype[name] = 'global' if isglobal else 'perpoll'

def ppp_pymol_get(name):
    if not ppppp: return TEST_STATE[name]
    print('ppp get', state[f'ppp_pymol_{name}'])
    return state[f'ppp_pymol_{name}']

def ppp_pymol_set(name, val):
    # print('PYMOLSET', name, val, ppppp)
    if not ppppp: TEST_STATE[name] = val
    else: state[f'ppp_pymol_{name}'] = val

def widget_gettext(widget):
    if hasattr(widget, 'text'): return widget.text()
    if hasattr(widget, 'toPlainText'): return widget.toPlainText()
    if hasattr(widget, 'currentText'): return widget.currentText()

class StateManager:
    def __init__(self, config_file, state_file):
        self._statetype = dict(
            activepoll='global',
            polls='global',
            active_cmds='perpoll',
            reviewed='global',
            pymol_view='perpoll',
        )
        self.config_file, self.state_file = config_file, state_file
        self.load()

    def load(self):
        self.local = self.read_config(self.config_file, opts={}, cmds={})
        self._localstate = self.read_config(
            self.state_file,
            strict=False,
            activepoll=None,
            active_cmds=set(),
            pymol_view={},
        )
        self.polls = self._localstate.polls
        assert not self.polls.__dict__['_special']['strict_lookup']
        assert self.polls.__dict__['_special']['default'] == 'bunchwithparent'

    def read_config(self, fname, strict=True, **kw):
        result = ipd.dev.Bunch(**kw)
        if os.path.exists(fname):
            with open(fname) as inp:
                result |= ipd.Bunch(yaml.load(inp, yaml.CLoader))
        return ipd.dev.make_autosave_hierarchy(result,
                                               _strict=strict,
                                               _autosave=fname,
                                               _default='bunchwithparent')

    def save(self):
        self.local._notify_changed()
        self._localstate._notify_changed()

    def is_global_state(self, name):
        return 'global' == self._statetype[name]

    def is_per_poll_state(self, name):
        return 'perpoll' == self._statetype[name]

    def is_pymol_state(self, name):
        return 'pymol' == self._statetype[name]

    def set_state_type(self, name, statetype):
        assert name not in self._statetype or self._statetype[name] == statetype
        self._statetype[name] = statetype

    def __contains__(self, name):
        return self._haveperpolstate(name)

    def get_per_poll(self, name, poll):
        return self._getperpollstate(name, poll)

    def get(self, name):
        assert name != 'polls'
        if name in _debug_state: print(f'GET {name} type {self._statetype[name]}')
        if self.is_global_state(name):
            if name not in self.local.opts and name in DEFAULTS:
                if name in _debug_state: print(f'set default {name} to self.local.opts')
                setattr(self.local.opts, name, DEFAULTS[name])
            if name not in self.local.opts:
                if name in _debug_state: print(f'get {name} from self._localstate')
                return self._localstate[name]
            if name in _debug_state: print(f'get {name} from self.local.opts')
            return self.local.opts[name]
        assert self.is_per_poll_state(name)
        return self._getperpollstate(name)

    def set(self, name, val):
        if self.is_global_state(name):
            with contextlib.suppress(ValueError):
                self.get(name)
            if name in self.local.opts:
                if name in _debug_state: print(f'set {name} in self.local.opts')
                return setattr(self.local.opts, name, val)
            else:
                if name in _debug_state: print(f'set {name} in self._localstate')
                return setattr(self._localstate, name, val)
        return self._setperpollstate(name, val)

    __getitem__ = get
    __getattr__ = get
    __setitem__ = set

    # __setattr__ = set

    def _haveperpolstate(self, name):
        if ppppp and ppppp.polls.pollinprogress and name in self._localstate.polls[
                ppppp.polls.pollinprogress.poll.name]:
            return True
        if name in self._localstate: return True
        if name in self.local.opts: return True
        return False

    def _getperpollstate(self, name, poll=None, indent=''):
        poll = ppppp.polls.pollinprogress.poll.name if ppppp.polls.pollinprogress else poll
        if poll and name in self._localstate.polls[poll]:
            if name in _debug_state: print('Get', name, self._localstate.polls[poll][name], 'from poll', poll)
            return self._localstate.polls[poll][name]
        else:
            if name in self._localstate:
                val = self._localstate[name]
                if name in _debug_state: print(f'{indent}Get', name, val, 'from self._localstate', poll)
            elif name in self.local.opts:
                val = self.local.opts[name]
                if name in _debug_state: print(f'{indent}Get', name, val, 'from opts', poll)
            elif name in DEFAULTS:
                val = DEFAULTS[name]
                if name in _debug_state: print(f'{indent}Get', name, val, 'from defaults', poll)
            else:
                raise ValueError(f'unknown self._localstate {name}')
            if poll:
                if name in _debug_state: print('SET perpoll', name, '=', val)
                setattr(self._localstate.polls[poll], name, val)
            return val
        raise ValueError(findent + 'Get unknown self._localstate {name}')

    def _setperpollstate(self, name, val, poll=None):
        with contextlib.suppress(ValueError):
            self._getperpollstate(name, poll, indent='   ')  # check already exists
        poll = ppppp.polls.pollinprogress.poll.name if ppppp.polls.pollinprogress else poll
        if poll:
            if name in _debug_state: print('Set', name, val, 'topoll', poll)
            dest = self._localstate.polls[poll]
        elif name in self.local.opts and name not in self._localstate:
            dest = self.local.opts
            if name in _debug_state: print('Set', name, val, 'topots', poll)
        else:
            dest = self._localstate
            if name in _debug_state: print('Set', name, val, 'tostate', poll)
        setattr(dest, name, val)

class SubjectName:
    def __init__(self):
        self.count = 0
        self.name = 'subject'

    def __call__(self):
        return f'{self.name}_{self.count}'

    def new(self, name='subject'):
        self.count += 1
        for sfx in ppp.STRUCTURE_FILE_SUFFIX:
            name = name.replace(sfx, '')
        self.name = os.path.basename(name)
        return self()

subject_name = SubjectName()

@profile
class PymolFileViewer:
    def __init__(self, fname, name, toggles):
        self.fname = fname
        self.toggles = toggles
        pymol.cmd.delete(subject_name())
        pymol.cmd.load(self.fname, subject_name.new(name))
        pymol.cmd.color('green', f'{subject_name()} and elem C')
        self.update()

    def update(self):
        pymol.cmd.reset()
        for cmd in state.active_cmds:
            assert cmd in ppppp.toggles.cmds
            self.run_command(ppppp.toggles.cmds[cmd].cmdon)
        if self.fname in (pview := state.pymol_view): pymol.cmd.set_view(pview[self.fname])

    def update_toggle(self, toggle: 'ToggleCommand'):
        if toggle: self.run_command(toggle.cmdon)
        else: self.run_command(toggle.cmdoff)

    def run_command(self, cmd: str):
        assert isinstance(cmd, str)
        pymol.cmd.do(cmd.replace('$subject', subject_name()))

    def cleanup(self):
        pymol.cmd.delete(subject_name())
        state.pymol_view[self.fname] = pymol.cmd.get_view()

@profile
class FileFetcher(threading.Thread):
    def __init__(self, fname, cache):
        super().__init__()
        self.fname = fname
        self.localfname = cache.tolocal(fname)
        self.tmpfname = f'{self.localfname}.tmp'
        self.start()

    def run(self):
        shutil.copyfile(self.fname, self.tmpfname)
        shutil.move(self.tmpfname, self.localfname)

@profile
class FileCache:
    def __init__(self, fnames, **kw):
        self.fnames = fnames

    def __getitem__(self, i):
        return self.fnames[i]

    def cleanup(self):
        pass

@profile
class PrefetchLocalFileCache(FileCache):
    '''
    Copies files to a CONF temp directory. Will downloads files ahead of requested index in background.
    '''
    def __init__(self, fnames, numprefetch=7, path='/tmp/ppp/filecache'):
        self.path = path
        self.fetchers = {}
        os.makedirs(path, exist_ok=True)
        self.available = set(os.listdir(path))
        self.fnames = fnames
        self.numprefetch = numprefetch
        self[0]

    def update_fetchers(self):
        done = {k for k, v in self.fetchers.items() if not v.is_alive()}
        self.available |= done
        for k in done:
            del self.fetchers[k]

    def prefetch(self, fname):
        if isinstance(fname, list):
            assert all(self.prefetch(f) for f in fname)
        if fname in self.available or fname in self.fetchers: return True
        self.update_fetchers()
        if len(self.fetchers) > 10: return False
        self.fetchers[fname] = FileFetcher(fname, self)
        return True

    def tolocal(self, fname):
        slash = '\\'
        return f"{self.path}/{fname.replace('/',slash)}"

    def __getitem__(self, index):
        assert self.prefetch(self.fnames[index])
        for i in range(min(self.numprefetch, len(self.fnames) - index - 1)):
            self.prefetch(self.fnames[index + i + 1])
        localfname = self.tolocal(self.fnames[index])
        for _ in range(100):
            self.update_fetchers()
            if self.fnames[index] in self.available:
                return localfname
            time.sleep(0.1)
        isfalse_notify(os.path.exists(localfname))

    def cleanup(self):
        self.update_fetchers()
        for f in self.fetchers:
            f.killed = True

def fnames_from_path(fnames):
    if os.path.isdir(fnames):
        fnames = [os.path.join(fnames, _) for _ in os.listdir(fnames) if _.endswith(ppp.STRUCTURE_FILE_SUFFIX)]
    else:
        with open(fnames) as inp:
            fnames = list(map(os.path.abspath, map(str.strip, inp)))
        if not all(f.endswith(ppp.STRUCTURE_FILE_SUFFIX) or os.path.isdir(f) for f in fnames): return None
        if not all(os.path.exists(f) for f in fnames): return None
    return fnames

@profile
class PollInProgress:
    def __init__(self, poll):
        self.poll = poll
        self.viewer = None
        Cache = PrefetchLocalFileCache if state.prefetch else FileCache
        self.filecache = Cache(self.fnames, numprefetch=7 if state.prefetch else 0)
        ppppp.set_pbar(0, len(state.get_per_poll('reviewed', poll=self.poll.name)), len(self.fnames))
        ppppp.widget.showsym.setText(self.poll.sym)
        ppppp.widget.showlig.setText(self.poll.ligand)
        ppppp.toggles.update_toggles_gui()

    def init_files(self, fnames):
        if isinstance(fnames, (str, bytes)):
            fnames = fnames_from_path(fnames)
        # fnames = [_ for _ in fnames if _ not in state.polls.reviewed]
        if state.shuffle: self.pbdlist = random.shuffle(fnames)
        ppppp.set_pbar(lb=0, val=len(state.reviewed), ub=len(fnames) - 1)
        return fnames

    @property
    def fnames(self):
        if 'fnames' not in state.polls[self.poll.name]:
            state.polls[self.poll.name].fnames = self.init_files(self.poll.path)
        return state.polls[self.poll.name].fnames

    @property
    def index(self):
        if 'activepollindex' not in state.polls[self.poll.name]:
            state.polls[self.poll.name].activepollindex = 0
        return state.polls[self.poll.name].activepollindex

    @index.setter
    def index(self, index):
        state.polls[self.poll.name].activepollindex = index

    def start(self):
        ppppp.update_opts()
        self.switch_to(self.index)

    def switch_to(self, index=None, delta=None):
        if index is None: index = self.index
        if delta: index = (index + delta) % len(self.fnames)
        if index >= len(self.fnames): return False
        if self.viewer: self.viewer.cleanup()
        ppppp.widget.showfile.setText(self.fnames[index])
        self.viewer = PymolFileViewer(self.filecache[index], self.fnames[index], ppppp.toggles)
        self.index = index
        return True

    def record_review(self, grade, comment):
        review = ppp.ReviewSpec(grade=grade,
                                comment=comment,
                                polldbkey=self.poll.dbkey,
                                fname=self.fnames[self.index])
        if state.do_review_action and not self.exec_review_action(review): return
        response = remote.upload(review)
        if isfalse_notify(not response, f'server response: {response}'): return
        self.review_accepted(review)

    def review_accepted(self, review):
        pymol.cmd.delete(subject_name())
        state.reviewed.add(self.viewer.fname)
        ppppp.set_pbar(lb=0, val=len(state.reviewed), ub=len(self.fnames) - 1)
        if len(state.reviewed) == len(self.fnames): ppppp.polls.poll_finished()
        else: self.switch_to(delta=1)

    def preprocess_shell_cmd(self, cmd):
        cmd = cmd.replace('$pppdir', os.path.abspath(os.path.expanduser(state.pppdir)))
        cmd = cmd.replace('$poll', self.poll.name.replace(' ', '_'))
        cmd = cmd.replace('$filebase', os.path.basename(self.fnames[self.index]))
        cmd = cmd.replace('$file', self.viewer.fname)
        cmds = []
        for line in cmd.split(os.linesep):
            for c in line.split(';'):
                noopt = [_ for _ in c.split() if not _.startswith('-')]
                if noopt[0] in 'cp rsync'.split():
                    # print('MAKEDIR', os.path.expanduser(os.path.dirname(noopt[-1])))
                    os.makedirs(os.path.expanduser(os.path.dirname(noopt[-1])), exist_ok=True)
                    for fn in filter(lambda s: s.endswith(ppp.STRUCTURE_FILE_SUFFIX), noopt[1:-1]):
                        assert os.path.exists(fn)
                cmds.append(c.split())
        return cmds

    def exec_review_action(self, review):
        cmds = self.preprocess_shell_cmd(state.review_action.replace('$grade', review.grade))
        for cmd in cmds:
            try:
                result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                # print(result)
            except subprocess.CalledProcessError as e:
                msg = (f'on review action failed to execute:\n{" ".join(cmd)}\nRETURN CODE: '
                       f'{e.returncode}\nOUTPUT: {e.output.decode()}\nEXCEPTION: {e.cmd}')
                notify(msg)
                return False
        # return not isfalse_notify(not result, f'review action output: {result}')
        return True

    def cleanup(self):
        if self.viewer: self.viewer.cleanup()
        ppppp.widget.showfile.setText('')
        ppppp.widget.showsym.setText('')
        ppppp.widget.showlig.setText('')
        self.filecache.cleanup()

@profile
class Polls:
    def __init__(self):
        self.pollinprogress = None
        self.current_poll_index = None
        self.listitems = None

    def init_session(self, widget):
        self.widget = widget
        self.widget.itemClicked.connect(lambda a: self.poll_clicked(a))
        self.newpollwidget = pymol.Qt.QtWidgets.QDialog()
        uifile = os.path.join(os.path.dirname(__file__), 'gui_new_poll.ui')
        self.newpollwidget = pymol.Qt.utils.loadUi(uifile, self.newpollwidget)
        self.newpollwidget.openfiledialog.clicked.connect(lambda: self.open_file_picker())
        self.newpollwidget.cancel.clicked.connect(lambda: self.newpollwidget.hide())
        self.newpollwidget.autofill.clicked.connect(lambda: self.create_poll_autofill_button())
        self.newpollwidget.ok.clicked.connect(lambda: self.create_poll_ok_button())
        self.refresh_polls()

    def open_file_picker(self):
        dialog = pymol.Qt.QtWidgets.QFileDialog(self.newpollwidget)
        dialog.setFileMode(pymol.Qt.QtWidgets.QFileDialog.Directory)
        dialog.setDirectory(os.path.expanduser('~'))
        dialog.show()
        if dialog.exec_():
            file_names = dialog.selectedFiles()
            assert len(file_names) == 1
            self.newpollwidget.path.setText(file_names[0])

    def refresh_polls(self):
        # localpolls = [(p.dbkey, p.name, p.user, p.desc, p.sym, p.ligand) for p in state.local.polls.values()]
        self.pollsearchtext, self.polltooltip, allpolls = [], {}, {}
        self.allpolls = remote.pollinfo()  #+ localpolls
        for key, name, user, desc, sym, lig, nchain in self.allpolls:
            ttip = f'NAME: {name}\nDESCRIPTION: DBKEY:{key}\n{desc}\nSYM: {sym}\nUSER: {user}\nLIG: {lig}\nNCHAIN: {nchain}'
            self.polltooltip[name] = ttip
            self.pollsearchtext.append(f'{name}||||{desc} sym:{sym} user:{user} lig:{lig}')
            allpolls[name] = key
        self.allpolls = allpolls
        self.pollsearchtext = '\n'.join(self.pollsearchtext)
        self.widget.clear()
        if self.widget.count() == 0:
            self.listitems, self.listitemdict = [], {}
            for i, name in enumerate(sorted(self.allpolls)):
                self.widget.addItem(name)
                self.listitems.append(self.widget.item(i))
                self.listitems[-1].setToolTip(self.polltooltip[name])
                self.listitemdict[name] = self.listitems[-1]
        self.update_polls_gui()
        if state.activepoll and state.activepoll in self.allpolls:
            self.poll_start(state.activepoll)

    def filtered_poll_list(self):
        polls = set(self.allpolls)
        if query := state.findpoll:
            from subprocess import Popen, PIPE
            p = Popen(['fzf', '-i', '--filter', f'{query}'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
            hits = p.communicate(input=self.pollsearchtext)[0]
            hits = [m[:m.find('||||')] for m in hits.split('\n') if m]
            polls = set(hits)
            if self.pollinprogress: polls.add(self.pollinprogress.poll.name)
        return polls

    def update_polls_gui(self):
        if self.listitems is None: self.refresh_polls()
        self.visiblepolls = self.filtered_poll_list()
        if state.activepoll in self.listitemdict:
            self.listitemdict[state.activepoll].setSelected(True)
        for item in self.listitems:
            hidden = item.text() not in self.visiblepolls and not item.isSelected()
            item.setHidden(hidden)

    def pollstatus(self, poll):
        if not os.path.exists(poll.path): return 'invalid'
        if os.path.getsize(poll.path) == 0: return 'invalid'
        if poll.ispublic: return 'public'
        return 'private'

    def poll_clicked(self, item):
        assert item.isSelected()
        if item.isSelected():
            self.poll_start(item.text())
        else:
            # print('poll finished', item.text())
            self.poll_finished()

    def poll_start(self, name):
        assert name in self.allpolls
        if self.pollinprogress: self.pollinprogress.cleanup()
        poll = remote.poll(self.allpolls[name])
        self.pollinprogress = PollInProgress(poll)
        state.activepoll = self.pollinprogress.poll.name
        ppppp.toggles.update_toggles_gui()
        self.pollinprogress.start()

    def poll_finished(self):
        if self.pollinprogress: self.pollinprogress.cleanup()
        self.pollinprogress = None
        state.activepoll = None
        ppppp.update_opts()
        ppppp.set_pbar(done=True)
        # self.update_polls_gui()

    def create_poll_start(self):
        self.newpollwidget.show()

    def create_poll_spec_from_gui(self):
        # sourcery skip: dict-assign-update-to-union
        duration = ipd.dev.safe_eval('dict(' + ','.join(self.newpollwidget.duration.text().split()) + ')')
        duration = datetime.timedelta(**duration)
        duration = duration or datetime.timedelta(weeks=99999)
        if isfalse_notify(self.newpollwidget.name.text(), 'Must provide a Name'): return
        if isfalse_notify(os.path.exists(self.newpollwidget.path.text()), 'path must exist'): return
        if isfalse_notify(duration > datetime.timedelta(minutes=1), 'Poll expires too soon'): return
        fields = 'name path sym ligand user workflow cmdstart cmdstop props attrs'
        kw = {k: widget_gettext(getattr(self.newpollwidget, k)) for k in fields.split()}
        kw |= {k: bool(getattr(self.newpollwidget, k).checkState()) for k in 'ispublic telemetry'.split()}
        kw['enddate'] = datetime.datetime.now() + duration
        return self.create_poll_spec(**kw)

    def create_poll_autofill_button(self):
        poll = self.create_poll_spec_from_gui()
        for k in 'name path sym ligand user nchain'.split():
            getattr(self.newpollwidget, k).setText(str(poll[k]))

    def create_poll_ok_button(self):
        poll = self.create_poll_spec_from_gui()
        if self.create_poll(poll):
            self.newpollwidget.hide()

    def create_poll(self, poll: ppp.PollSpec, temporary=False):
        if not poll: return False
        response = remote.upload(poll)
        if isfalse_notify(not response, f'server response: {response}'): return False
        self.refresh_polls()
        return True

    def create_poll_spec(self, **kw):
        try:
            return ppp.PollSpec(**kw)
        except Exception as e:
            notify(f'create PollSpec error:\n{e}')
            return None

    def create_poll_from_curdir(self):
        u = getpass.getuser()
        d = os.path.abspath('.').replace(f'/mnt/home/{u}', '~').replace(f'/home/{u}', '~')
        self.create_poll(
            self.create_poll_spec(
                name=f"Files in {d.replace('/',' ')} ({u})",
                desc='The poll for lazy people',
                path=d,
                ispublic=False,
                telemetry=False,
                start=None,
                end=None,
                _temporary=True,
            ))

    def cleanup(self):
        if self.pollinprogress: self.pollinprogress.cleanup()

@profile
class ToggleCommand(ppp.PymolCMD):
    def __init__(self, widget, **kw):
        super().__init__(remote, **kw)
        self._widget = widget
        if self.cmdstart: pymol.cmd.do(self.cmdstart)
        assert self.widget.text().startswith(self.name)
        # self.widget.setCheckState(2 * int(self.onstart))

    @property
    def widget(self):
        return self._widget

    def widget_update(self, toggle=False):
        if toggle:
            self.widget.setCheckState(0 if self.widget.checkState() else 2)
            if self.widget.checkState(): state.active_cmds.add(self.name)
            else: state.active_cmds.remove(self.name)
        if ppppp.polls.pollinprogress and ppppp.polls.pollinprogress.viewer:
            print('toggle update')
            ppppp.polls.pollinprogress.viewer.update_toggle(toggle=self)

    def __bool__(self):
        return bool(self.widget.checkState())

@profile
class ToggleCommands:
    def __init__(self):
        self.widget = None
        self.itemsdict = None
        self.cmds = {}

    def init_session(self, widget):
        self.widget = widget
        self.refersh_toggle_list()
        # widget.itemChanged.connect(lambda _: self.update_item(_))
        self.widget.itemClicked.connect(lambda _: self.update_item(_, toggle=True))
        self.gui_new_pymolcmd = pymol.Qt.QtWidgets.QDialog()
        self.gui_new_pymolcmd = pymol.Qt.utils.loadUi(
            os.path.join(os.path.dirname(__file__), 'gui_new_pymolcmd.ui'), self.gui_new_pymolcmd)
        self.gui_new_pymolcmd.cancel.clicked.connect(lambda: self.gui_new_pymolcmd.hide())
        self.gui_new_pymolcmd.ok.clicked.connect(lambda: self.create_toggle_done())

    def update_item(self, item, toggle=False):
        self.cmds[item.text()].widget_update(toggle)

    def create_toggle_start(self):
        self.gui_new_pymolcmd.show()

    def create_toggle_done(self):  # sourcery skip: dict-assign-update-to-union
        if isfalse_notify(self.gui_new_pymolcmd.name.text(), 'Must provide a Name'): return
        if isfalse_notify(self.gui_new_pymolcmd.cmdon.toPlainText(), 'Must provide a command'): return
        fields = 'name cmdon cmdoff cmdstart sym ligand props attrs'
        kw = {k: widget_gettext(getattr(self.gui_new_pymolcmd, k)) for k in fields.split()}
        kw |= {k: bool(getattr(self.gui_new_pymolcmd, k).checkState()) for k in 'ispublic onstart'.split()}
        cmdspec = ppp.PymolCMDSpec(**kw)
        if isfalse_notify(not cmdspec.errors(), cmdspec.errors()): return
        self.gui_new_pymolcmd.hide()
        if cmdspec.ispublic:
            result = remote.upload(cmdspec)
            assert not result, result
        else:
            cmd = ppp.PymolCMD(None, dbkey=len(state.local.cmds) + 1, **cmdspec.dict())
            setattr(state.local.cmds, cmd.name, cmd)
        self.refersh_toggle_list()
        self.update_toggles_gui()

    def refersh_toggle_list(self):
        assert self.widget is not None
        cmdsdicts = list(state.local.cmds.values()) + remote.pymolcmdsdict()
        # print([c['name'] for c in cmdsdicts])
        if 'active_cmds' not in state:
            state.active_cmds = {cmd['name'] for cmd in cmdsdict if cmd['onstart']}
        self.itemsdict = {}
        self.widget.clear()
        for cmd in cmdsdicts:
            self.widget.addItem(cmd['name'])
            item = self.widget.item(self.widget.count() - 1)
            item.setFlags(item.flags() | pymol.Qt.QtCore.Qt.ItemIsUserCheckable)
            self.itemsdict[cmd['name']] = item
            cmd = ToggleCommand(item, **cmd)
            item.setToolTip(
                f'NAME: {cmd.name}\nON: {cmd.cmdon}\nOFF: {cmd.cmdoff}\nNCHAIN: {cmd.minchains}-{cmd.maxchains}'
                f'\nispublic: {cmd.ispublic}\nSYM: {cmd.sym}\nLIG:{cmd.ligand}\nDBKEY:{cmd.dbkey}')
            cmd.widget.setCheckState(2) if cmd.name in state.active_cmds else cmd.widget.setCheckState(0)
            self.cmds[cmd.name] = cmd
        self.cmdsearchtext = '\n'.join(f'{c.name}||||{c.desc} sym:{c.sym} user:{c.user} lig:{c.ligand}'
                                       for c in self.cmds.values())
        self.update_toggles_gui()

    def filtered_cmd_list(self):
        hits = set(self.cmds.keys())
        if query := state.findcmd:
            from subprocess import Popen, PIPE
            p = Popen(['fzf', '-i', '--filter', f'{query}'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
            hits = p.communicate(input=self.cmdsearchtext)[0]
            hits = [m[:m.find('||||')] for m in hits.split('\n') if m]
        cmds, sym, ligand, nchain = self.cmds, '', '', -1
        if not state.showallcmds and len(hits) < 100:
            if pip := ppppp.polls.pollinprogress:
                sym, ligand, nchain = pip.poll.sym, pip.poll.ligand, pip.poll.nchain
            hits = filter(lambda x: cmds[x].sym in ('', sym), hits)
            hits = filter(lambda x: cmds[x].ligand in ligand or (ligand and cmds[x].ligand == 'ANY'), hits)
            if nchain > 0: hits = filter(lambda x: cmds[x].minchains <= nchain <= cmds[x].maxchains, hits)
        return set(hits) | state.active_cmds

    def update_toggles_gui(self):
        if self.itemsdict is None: self.refersh_toggle_list()
        visible = {k: self.cmds[k] for k in self.filtered_cmd_list()}
        for name, item in self.itemsdict.items():
            item.setCheckState(2 if name in state.active_cmds else 0)
            item.setHidden(not (name in visible or item.isSelected()))

    def cleanup(self):
        pass

@profile
class PrettyProteinProjectPymolPluginPanel:
    def init_session(self):
        self.polls = Polls()
        self.toggles = ToggleCommands()
        pymol.cmd.save(SESSION_RESTORE)
        self.setup_main_window()
        self.update_opts()
        self.toggles.init_session(self.widget.toggles)
        self.polls.init_session(self.widget.polls)

    def setup_main_window(self):
        self.widget = pymol.Qt.QtWidgets.QDialog()
        self.widget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'gui_grid_main.ui'),
                                            self.widget)
        self.widget.show()
        for grade in 'SABCDF':
            getattr(self.widget, f'{grade.lower()}tier').clicked.connect(partial(self.grade_pressed, grade))
        self.widget.button_newpoll.clicked.connect(lambda: self.polls.create_poll_start())
        self.widget.button_use_curdir.clicked.connect(lambda: self.polls.create_poll_from_curdir())
        self.widget.button_refresh_polls.clicked.connect(lambda: self.polls.refresh_polls())
        self.widget.button_refresh_cmds.clicked.connect(lambda: self.toggles.refersh_toggle_list())
        self.widget.button_newopt.clicked.connect(lambda: self.toggles.create_toggle_start())
        self.widget.button_save.clicked.connect(lambda: self.save_session())
        self.widget.button_load.clicked.connect(lambda: self.load_session())
        self.widget.button_restart.clicked.connect(lambda: self.init_session())
        self.widget.button_quit.clicked.connect(lambda: self.quit())
        self.widget.button_quitpymol.clicked.connect(lambda: self.quit(exitpymol=True))
        self.keybinds = []
        self.add_keybind('pgup', 'LeftArrow', lambda: self.polls.pollinprogress.switch_to(delta=-1))
        self.add_keybind('pgdn', 'RightArrow', lambda: self.polls.pollinprogress.switch_to(delta=1))

    def add_keybind(self, key, qkey, action):
        if qkey:  # Qt binds not workind for some reason
            keybind = pymol.Qt.QtWidgets.QShortcut(qkey, self.widget)
            keybind.activated.connect(action)
            self.keybinds.append(keybind)
        pymol.cmd.set_key(key, action)

    def update_opts(self):
        # print('UPDATE OPTS', ppppp.polls.pollinprogress)
        action = collections.defaultdict(lambda: lambda: None)
        action['hide_invalid'] = self.polls.update_polls_gui
        action['showallcmds'] = self.toggles.update_toggles_gui
        action['findpoll'] = self.polls.update_polls_gui
        action['findcmd'] = self.toggles.update_toggles_gui
        for name, widget in self.widget.__dict__.items():
            if name.startswith('opt_'): opt, statetype = name[4:], 'perpoll'
            elif name.startswith('globalopt_'): opt, statetype = name[10:], 'global'
            else: continue
            state.set_state_type(opt, statetype)
            if widget.__class__.__name__.endswith('QLineEdit'):
                if opt in state: widget.setText(state[opt])
                else: state.set(opt, widget.text())
                widget.textChanged.connect(lambda _, opt=opt: (state.set(opt, _), action[opt]()))
            elif widget.__class__.__name__.endswith('QCheckBox'):
                if opt in state: widget.setCheckState(state[opt])
                else: state.set(opt, 2 if widget.checkState() else 0)
                widget.stateChanged.connect(lambda _, opt=opt: (state.set(opt, _), action[opt]()))
            else:
                raise ValueError(f'dont know how to use option widget type {type(v)}')

    def grade_pressed(self, grade):
        if isfalse_notify(self.polls.pollinprogress, 'No active poll!'): return
        if self.polls.pollinprogress.record_review(grade, comment=self.widget.comment.toPlainText()):
            self.widget.comment.clear()

    def set_pbar(self, lb=None, val=None, ub=None, done=None):
        if done: return self.widget.progress.setProperty('enabled', False)
        self.widget.progress.setProperty('enabled', True)
        if lb is not None: self.widget.progress.setMinimum(lb)
        if ub is not None: self.widget.progress.setMaximum(ub)
        if val is not None: self.widget.progress.setValue(val)

    def save_session(self):
        state.save()

    def load_session(self):
        state.load()
        self.init_session()

    def quit(self, exitpymol=False):
        self.widget.hide()
        self.polls.cleanup()
        self.toggles.cleanup()
        self.save_session()
        if os.path.exists(SESSION_RESTORE):
            pymol.cmd.load(SESSION_RESTORE)
            os.remove(SESSION_RESTORE)
        ipd.dev.global_timer.report()
        if exitpymol: sys.exit()

@profile
def run_local_server(port=54321):
    ipd.dev.lazyimport('fastapi')
    ipd.dev.lazyimport('sqlmodel')
    ipd.dev.lazyimport('uvicorn')
    ipd.dev.lazyimport('ordered_set')
    args = dict(port=port, log='warning', datadir=os.path.expanduser('~/.config/ppp/localserver'), local=True)
    __server_thread = threading.Thread(target=ppp.server.run, kwargs=args, daemon=True)
    __server_thread.start()
    # dialog should cover server start time
    isfalse_notify(False, f"Can't connt to: {SERVER_ADDR}\nWill try to run a local server.")
    return ppp.PPPClient(f'127.0.0.1:{port}')

@profile
def run(_self=None):
    os.makedirs(os.path.dirname(SESSION_RESTORE), exist_ok=True)
    os.makedirs(os.path.dirname(PPPPP_PICKLE), exist_ok=True)
    if os.path.exists(SESSION_RESTORE): os.remove(SESSION_RESTORE)
    pymol.cmd.do('from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin '
                 'import ppp_pymol_get, ppp_pymol_set, ppp_pymol_add_default')
    global ppppp, remote, state
    state = StateManager(CONFIG_FILE, STATE_FILE)
    try:
        remote = ppp.PPPClient(SERVER_ADDR)
    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectionError):
        remote = run_local_server()
    ppppp = PrettyProteinProjectPymolPluginPanel()
    ppppp.init_session()

def main():
    print('RUNNING main if for debugging only!')

    while True:
        sleep(0.1)

if __name__ == '__main__':
    main()
