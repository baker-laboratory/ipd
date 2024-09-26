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
import subprocess
import sys
import tempfile
import threading
import time
import traceback

it = ipd.lazyimport('itertools', 'more_itertools', pip=True)
requests = ipd.lazyimport('requests', pip=True)
fuzzyfinder = ipd.lazyimport('fuzzyfinder', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)

remote, conf, ppppp = None, None, None
SERVER_ADDR = os.environ.get('PPPSERVER', '127.0.0.1:12345')
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
                pymol_sc_repr='sticks')
NOT_PER_POLL = {'prefetch'}
# profile = ipd.dev.timed
profile = lambda f: f
_debug_state = {'pppdir'}

def notify(message):
    pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)

def isfalse_notify(ok, message):
    if not ok:
        pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)
        return True

def ppp_pymol_add_default(name, val):
    DEFAULTS[f'pymol_{name}'] = val

def ppp_pymol_get(name):
    if not ppppp: return TEST_STATE[name]
    return getstate(f'pymol_{name}')

def ppp_pymol_set(name, val):
    print('PYMOLSET', name, val, ppppp)
    if not ppppp:
        TEST_STATE[name] = val
    else:
        return setstate(f'pymol_{name}', val)

def haveconf(k):
    return k in conf.opt

def getconf(k):
    return conf.opt[k]

def setconf(k, v):
    return setattr(conf.opt, k, v)

def havestate(name):
    if ppppp and ppppp.polls.pollinprogress and name in state.polls[ppppp.polls.pollinprogress.poll.name]:
        return True
    if name in state: return True
    if name in conf.opts: return True
    return False

def getstate(name, poll=None, indent=''):
    poll = ppppp.polls.pollinprogress.poll.name if ppppp.polls.pollinprogress else poll
    if name not in NOT_PER_POLL and poll and name in state.polls[poll]:
        if name in _debug_state: print('Get', name, state.polls[poll][name], 'from poll', poll)
        return state.polls[poll][name]
    else:
        if name in state:
            val = state[name]
            if name in _debug_state: print(f'{indent}Get', name, val, 'from state', poll)
        elif name in conf.opts:
            val = conf.opts[name]
            if name in _debug_state: print(f'{indent}Get', name, val, 'from opts', poll)
        elif name in DEFAULTS:
            val = DEFAULTS[name]
            if name in _debug_state: print(f'{indent}Get', name, val, 'from defaults', poll)
        else:
            raise ValueError(f'unknown state {name}')
        if poll:
            print(name)
            print('    set poll val', val)
            setattr(state.polls[poll], name, val)
        return val
    raise ValueError(findent + 'Get unknown state {name}')

def setstate(name, val, poll=None):
    with contextlib.suppress(ValueError):
        getstate(name, poll, indent='   ')  # check already exists
    poll = ppppp.polls.pollinprogress.poll.name if ppppp.polls.pollinprogress else poll
    if name not in NOT_PER_POLL and poll:
        if name in _debug_state: print('Set', name, val, 'topoll', poll)
        dest = state.polls[poll]
    elif name in conf.opts and name not in state:
        dest = conf.opts
        if name in _debug_state: print('Set', name, val, 'topots', poll)
    else:
        dest = state
        if name in _debug_state: print('Set', name, val, 'tostate', poll)
    setattr(dest, name, val)

_subject_count = 0

def subject_name():
    return 'subject%i' % _subject_count

def new_subject_name():
    global _subject_count
    _subject_count += 1
    return subject_name()

@profile
class PymolFileViewer:
    def __init__(self, fname, toggles):
        self.fname = fname
        self.init_session(toggles)

    def init_session(self, toggles):
        self.toggles = toggles
        pymol.cmd.delete(subject_name())
        pymol.cmd.load(self.fname, new_subject_name())
        pymol.cmd.color('green', f'{subject_name()} and elem C')
        self.update()

    def update(self):
        pymol.cmd.reset()
        for cmd in getstate('active_cmds'):
            assert cmd in ppppp.toggles.cmdmap
            self.run_command(ppppp.toggles.cmdmap[cmd].cmdon)
        if self.fname in (pview := getstate('pymol_view')): pymol.cmd.set_view(pview[self.fname])

    def update_toggle(self, toggle: 'ToggleCommand'):
        if toggle: self.run_command(toggle.cmdon)
        else: self.run_command(toggle.cmdoff)

    def run_command(self, cmd: str):
        assert isinstance(cmd, str)
        pymol.cmd.do(cmd.replace('$subject', subject_name()))

    def cleanup(self):
        pymol.cmd.delete(subject_name())
        getstate('pymol_view')[self.fname] = pymol.cmd.get_view()

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
        print('Basic FileCache')
        self.fnames = fnames

    def __getitem__(self, i):
        return self.fnames[i]

    def cleanup(self):
        pass

@profile
class PrefetchLocalFileCache(FileCache):
    '''
    Copies files to a conf temp directory. Will downloads files ahead of requested index in background.
    '''
    def __init__(self, fnames, numprefetch=7, path='/tmp/ppp/filecache'):
        print('PrefetchLocalFileCache')
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
        Cache = PrefetchLocalFileCache if getstate('prefetch') else FileCache
        self.filecache = Cache(self.fnames, numprefetch=7 if conf.prefetch else 0)
        ppppp.set_pbar(0, len(getstate('reviewed', poll=self.poll.name)), len(self.fnames))
        ppppp.widget.showsym.setText(self.poll.sym)
        ppppp.widget.showlig.setText(self.poll.ligand)

    def init_files(self, fnames):
        if isinstance(fnames, (str, bytes)):
            fnames = fnames_from_path(fnames)
        # fnames = [_ for _ in fnames if _ not in state.polls.reviewed]
        if conf.opts.shuffle: self.pbdlist = random.shuffle(fnames)
        ppppp.set_pbar(lb=0, val=len(getstate('reviewed')), ub=len(fnames) - 1)
        return fnames

    # def init_session(self):
    # if self.viewer: self.viewer.init_session(ppppp.toggles)

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
        self.viewer = PymolFileViewer(self.filecache[index], ppppp.toggles)
        self.index = index
        return True

    def record_review(self, grade, comment):
        review = ppp.ReviewSpec(grade=grade,
                                comment=comment,
                                polldbkey=self.poll.dbkey,
                                fname=self.fnames[self.index])
        if getstate('do_review_action') and not self.exec_review_action(review): return
        response = remote.upload(review)
        if isfalse_notify(not response, f'server response: {response}'): return
        self.review_accepted(review)

    def review_accepted(self, review):
        pymol.cmd.delete(subject_name())
        getstate('reviewed').add(self.viewer.fname)
        ppppp.set_pbar(lb=0, val=len(getstate('reviewed')), ub=len(self.fnames) - 1)
        if len(getstate('reviewed')) == len(self.fnames): ppppp.polls.poll_finished()
        else: self.switch_to(delta=1)

    def preprocess_shell_cmd(self, cmd):
        cmd = cmd.replace('$pppdir', os.path.abspath(os.path.expanduser(conf.opt.pppdir)))
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
        cmds = self.preprocess_shell_cmd(getstate('review_action').replace('$grade', review.grade))
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

    def init_session(self, listwidget):
        self.listwidget = listwidget
        self.listwidget.itemClicked.connect(lambda a: self.poll_clicked(a))
        self.newpollwidget = pymol.Qt.QtWidgets.QDialog()
        uifile = os.path.join(os.path.dirname(__file__), 'gui_new_poll.ui')
        self.newpollwidget = pymol.Qt.utils.loadUi(uifile, self.newpollwidget)
        self.newpollwidget.openfiledialog.clicked.connect(lambda: self.open_file_picker())
        self.newpollwidget.cancel.clicked.connect(lambda: self.newpollwidget.hide())
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
        self.allpolls = remote.pollinfo() + [(p.dbkey, p.name, p.user, p.desc, p.sym, p.ligand)
                                             for p in conf.polls]
        self.pollsearch, self.polldesc, allpolls = [], {}, {}
        for k, n, u, d, s, l in self.allpolls:
            self.polldesc[n] = f'NAME: {n}\nDESCRIPTION: {d}\nSYM: {s}\nUSER: {u}\nLIG: {l}'
            self.pollsearch.append(f'{n}||||{d} sym:{s} user:{u} lig:{l}')
            allpolls[n] = k
        self.allpolls = allpolls
        self.pollsearch = '\n'.join(self.pollsearch)
        self.listwidget.clear()
        if self.listwidget.count() == 0:
            self.listitems, self.listitemdict = [], {}
            for i, name in enumerate(sorted(self.allpolls)):
                self.listwidget.addItem(name)
                self.listitems.append(self.listwidget.item(i))
                self.listitems[-1].setToolTip(self.polldesc[name])
                self.listitemdict[name] = self.listitems[-1]
        self.update_poll_list()
        if state.activepoll and state.activepoll in self.allpolls:
            self.poll_start(state.activepoll)

    def filtered_poll_list(self):
        polls = set(self.allpolls)
        if query := getstate('findpoll'):
            from subprocess import Popen, PIPE
            p = Popen(['fzf', '-i', '--filter', f'{query}'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
            hits = p.communicate(input=self.pollsearch)[0]
            hits = [m[:m.find('||||')] for m in hits.split('\n') if m]
            polls = set(hits)
            if self.pollinprogress: polls.add(self.pollinprogress.poll.name)
        return polls

    def update_poll_list(self):
        if not self.listitems: self.refresh_polls()
        self.visiblepolls = self.filtered_poll_list()
        if state.activepoll in self.listitemdict:
            self.listitemdict[state.activepoll].setSelected(True)
        for item in self.listitems:
            hidden = item.text() not in self.visiblepolls and not item.isSelected()
            item.setHidden(hidden)

    def pollstatus(self, poll):
        if not os.path.exists(poll.path): return 'invalid'
        if os.path.getsize(poll.path) == 0: return 'invalid'
        if poll['public']: return 'public'
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
        self.pollinprogress = PollInProgress(remote.poll(self.allpolls[name]))
        state.activepoll = self.pollinprogress.poll.name
        ppppp.toggles.update_toggle_list()
        self.pollinprogress.start()

    def poll_finished(self):
        if self.pollinprogress: self.pollinprogress.cleanup()
        self.pollinprogress = None
        state.activepoll = None
        ppppp.update_opts()
        ppppp.set_pbar(done=True)
        # self.update_poll_list()

    def create_poll_start(self):
        self.newpollwidget.show()

    def create_poll_ok_button(self):
        duration = ipd.dev.safe_eval('dict(' + ','.join(self.newpollwidget.duration.text().split()) + ')')
        duration = datetime.timedelta(**duration)
        duration = duration or datetime.timedelta(weeks=99999)
        if isfalse_notify(self.newpollwidget.name.text(), 'Must provide a Name'): return
        if isfalse_notify(self.newpollwidget.path.text(), 'Must provide a Path'): return
        if isfalse_notify(duration > datetime.timedelta(minutes=1), 'Poll expires too soon'): return
        poll = ppp.PollSpec(
            name=self.newpollwidget.name.text(),
            desc=self.newpollwidget.desc.text(),
            path=self.newpollwidget.path.text(),
            public=bool(self.newpollwidget.ispublic.checkState()),
            telem=bool(self.newpollwidget.telemetry.checkState()),
            start=datetime.datetime.strftime(datetime.datetime.now(), ppp.DATETIME_FORMAT),
            end=datetime.datetime.strftime(datetime.datetime.now() + duration, ppp.DATETIME_FORMAT),
        )
        if self.create_poll(poll):
            self.newpollwidget.hide()

    def create_poll(self, poll: ppp.PollSpec, temporary=False):
        poll.path = os.path.abspath(os.path.expanduser(poll.path))
        if isfalse_notify(os.path.exists(poll.path), 'path does not exist!'): return
        if isfalse_notify(fnames_from_path(poll.path),
                          f'contains no files with sufflix {ppp.STRUCTURE_FILE_SUFFIX}'):
            return
        if poll.public:
            response = remote.upload(poll)
            isfalse_notify(not response, f'server response: {response}')
        else:
            poll = ppp.Poll(**poll.dict(dbkey=f'conf{len(conf.polls)}'))
            setattr(conf.polls, poll.name, poll)
        self.refresh_polls()
        return True

    def create_poll_from_curdir(self):
        self.create_poll(
            ppp.PollSpec(
                name='Files in curent directory',
                desc='The poll for lazy people',
                path='.',
                public=False,
                telem=False,
                start=None,
                end=None,
                temporary=True,
            ))

    def cleanup(self):
        if self.pollinprogress: self.pollinprogress.cleanup()

@profile
class ToggleCommand(ppp.PymolCMD):
    def __init__(self, **kw):
        super().__init__(remote, **kw)
        self._status = 'public' if self.public else 'private'
        self._widget = None
        if self.cmdstart: pymol.cmd.do(self.cmdstart)

    @property
    def widget(self):
        return self._widget

    def init_session(self, widget):
        self._widget = widget
        assert self._widget.text().startswith(self.name)
        self._widget.setText(f'{self.name} ({self._status})')
        self._widget.setFlags(self._widget.flags() | pymol.Qt.QtCore.Qt.ItemIsUserCheckable)
        # self._widget.setCheckState(2 * int(self.onstart))

    def widget_update(self, toggle=False):
        if toggle:
            self._widget.setCheckState(0 if self._widget.checkState() else 2)
            if self._widget.checkState(): getstate('active_cmds').add(self.name)
            else: getstate('active_cmds').remove(self.name)
        if ppppp.polls.pollinprogress and ppppp.polls.pollinprogress.viewer:
            print('toggle update')
            ppppp.polls.pollinprogress.viewer.update_toggle(toggle=self)

    def __bool__(self):
        return bool(self._widget.checkState())

def copy_to_tempdir(tempdir, fname):
    basename = os.path.abspath(fname).replace('/', '\\')
    newname = os.path.join(tempdir, basename)
    return shutil.copyfile(fname, newname)

@profile
class ToggleCommands:
    def __init__(self):
        self.listwidget = None
        self.refresh_cmds()
        self.visible_cmds = self.cmds
        if not getstate('active_cmds'):
            setattr(conf.opts, 'active_cmds', {cmd.name for cmd in self.cmds if cmd.onstart})
            assert type(getstate('active_cmds')) != set

    def init_session(self, listwidget):
        self.listwidget = listwidget
        self.update_toggle_list()
        # listwidget.itemChanged.connect(lambda _: self.update_item(_))
        self.listwidget.itemClicked.connect(lambda _: self.update_item(_, toggle=True))
        self.gui_new_pymolcmd = pymol.Qt.QtWidgets.QDialog()
        self.gui_new_pymolcmd = pymol.Qt.utils.loadUi(
            os.path.join(os.path.dirname(__file__), 'gui_new_pymolcmd.ui'), self.gui_new_pymolcmd)
        self.gui_new_pymolcmd.cancel.clicked.connect(lambda: self.gui_new_pymolcmd.hide())
        self.gui_new_pymolcmd.ok.clicked.connect(lambda: self.create_toggle_done())

    def update_item(self, item, toggle=False):
        assert len(self.visible_cmds) == self.listwidget.count()
        self.visible_cmds[self.listwidget.indexFromItem(item).row()].widget_update(toggle)
        self.update_toggle_list()

    def create_toggle_start(self):
        self.gui_new_pymolcmd.show()

    def refresh_cmds(self):
        self.cmds = [ToggleCommand(**_) for _ in list(conf.cmds.values()) + remote.pymolcmdsdict()]
        self.cmdmap = {c.name: c for c in self.cmds}
        if self.listwidget: self.update_toggle_list()

    def create_toggle_done(self):
        cmd = ppp.PymolCMDSpec(
            name=self.gui_new_pymolcmd.name.text(),
            cmdon=self.gui_new_pymolcmd.cmdon.text(),
            cmdoff=self.gui_new_pymolcmd.cmdoff.text(),
            onstart=2 if self.gui_new_pymolcmd.onstart.checkState() else 0,
            public=2 if self.gui_new_pymolcmd.ispublic.checkState() else 0,
        )
        if isfalse_notify(not cmd.errors(), cmd.errors()):
            return
        self.gui_new_pymolcmd.hide()
        if self.cmds[-1].public:
            client.upload(cmdspec)
        else:
            setattr(conf.cmds, cmd.name, cmd)
        cmd = ToggleCommand(**cmd)
        self.cmds.append(cmd)
        self.listwidget.addItem(self.cmds[-1].name)
        item = self.listwidget.item(self.listwidget.count() - 1)
        item.setCheckState(0)
        self.cmds[-1].init_session(item)
        if item.checkState():
            getstate('active_cmds').add(item.text().lower().replace(' (public)', '').replace(' (private)', ''))

    def update_toggle_list(self):
        cmds = self.cmds
        if query := getstate('findcmd'):
            hits = set(fuzzyfinder.fuzzyfinder(query.lower(), [cmd.name.lower() for cmd in cmds]))
            hits |= {_.lower() for _ in getstate('active_cmds')}
            cmds = [cmd for cmd in self.cmds if cmd.name.lower() in hits]
        self.visible_cmds = cmds
        self.listwidget.clear()
        for cmd in cmds:
            self.listwidget.addItem(cmd.name)
            item = self.listwidget.item(self.listwidget.count() - 1)
            item.setToolTip(f'NAME: {cmd.name}\nON: {cmd.cmdon}\nOFF: {cmd.cmdoff}\n'
                            f'PUBLIC: {cmd.public}\nSYM: {cmd.sym}\nLIG:{cmd.ligand}')
            cmd.init_session(item)
            cmd.widget.setCheckState(2) if cmd.name in getstate('active_cmds') else cmd.widget.setCheckState(0)

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
        self.widget.button_refresh_cmds.clicked.connect(lambda: self.toggles.refresh_cmds())
        self.widget.button_newopt.clicked.connect(lambda: self.toggles.create_toggle_start())
        self.widget.button_save.clicked.connect(lambda: self.save_session())
        self.widget.button_load.clicked.connect(lambda: self.load_session())
        self.widget.button_restart.clicked.connect(lambda: self.init_session())
        self.widget.button_quit.clicked.connect(lambda: self.quit())
        self.widget.button_quitpymol.clicked.connect(lambda: self.quit(exitpymol=True))
        self.keybinds = []
        self.add_keybind('left', 'LeftArrow', lambda: self.polls.pollinprogress.switch_to(delta=-1))
        self.add_keybind('right', 'RightArrow', lambda: self.polls.pollinprogress.switch_to(delta=1))

    def add_keybind(self, key, qkey, action):
        if qkey:  # Qt binds not workind for some reason
            keybind = pymol.Qt.QtWidgets.QShortcut(qkey, self.widget)
            keybind.activated.connect(action)
            self.keybinds.append(keybind)
        pymol.cmd.set_key(key, action)

    def update_opts(self):
        # print('UPDATE OPTS', ppppp.polls.pollinprogress)
        action = collections.defaultdict(lambda: lambda: None)
        action['hide_invalid'] = self.polls.update_poll_list
        action['findpoll'] = self.polls.update_poll_list
        action['findcmd'] = self.toggles.update_toggle_list
        for name, widget in self.widget.__dict__.items():
            if name.startswith('opt_'):
                opt, have, get, set_ = name[4:], havestate, getstate, setstate
                # print('LOCALOPT', opt)
            elif name.startswith('globalopt_'):
                opt, have, get, set_ = name[10:], haveconf, getconf, setconf
                # print('GLOBALOPT', opt, have, get, set)
            else:
                continue
            if widget.__class__.__name__.endswith('QLineEdit'):
                if have(opt): widget.setText(get(opt))
                else: set_(opt, widget.text())
                widget.textChanged.connect(lambda _, opt=opt, setter=set_: (setter(opt, _), action[opt]()))
            elif widget.__class__.__name__.endswith('QCheckBox'):
                if have(opt): widget.setCheckState(get(opt))
                else: set_(opt, 2 if widget.checkState() else 0)
                widget.stateChanged.connect(lambda _, opt=opt, setter=set_: (setter(opt, _), action[opt]()))
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
        conf._notify_changed()
        state._notify_changed()

    def load_session(self):
        conf._autoreload_check()
        print(conf.cmds)
        state._autoreload_check()
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
    args = dict(port=port, log='warning', datadir=os.path.expanduser('~/.config/localserver'), local=True)
    __server_thread = threading.Thread(target=ppp.server.run, kwargs=args, daemon=True)
    __server_thread.start()
    # dialog should cover server start time
    isfalse_notify(False, f"Can't connt to: {SERVER_ADDR}\nWill try to run a local server.")
    return ppp.PPPClient(f'127.0.0.1:{port}')

@profile
def read_config(fname, **kw):
    result = ipd.dev.Bunch(**kw)
    if os.path.exists(fname):
        with open(fname) as inp:
            result |= ipd.Bunch(yaml.load(inp, yaml.CLoader))
    return ipd.dev.make_autosave_hierarchy(result, _autosave=fname, _default='bunchwithparent')

@profile
def run(_self=None):
    os.makedirs(os.path.dirname(SESSION_RESTORE), exist_ok=True)
    os.makedirs(os.path.dirname(PPPPP_PICKLE), exist_ok=True)
    if os.path.exists(SESSION_RESTORE): os.remove(SESSION_RESTORE)
    pymol.cmd.do(
        'from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin import ppp_pymol_get, ppp_pymol_set, ppp_pymol_add_default'
    )
    global ppppp, conf, state, remote
    conf = read_config(CONFIG_FILE, opts={}, cmds={}, polls={})
    state = read_config(STATE_FILE, activepoll=None, active_cmds=set(), cmds={}, polls={})
    for k in [k for k, v in conf.polls.items() if v.temporary]:
        del conf.polls[k]
    try:
        remote = ppp.PPPClient(SERVER_ADDR)
    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectionError):
        remote = run_local_server()
    #try:
    #    with open(PPPPP_PICKLE, 'rb') as inp:
    #        ppppp = pickle.load(inp)
    #        if not conf.opts.save_session: raise ValueError
    #        print('PrettyProteinProjectPymolPluginPanel LOADED FROM PICKLE')
    #except (FileNotFoundError, EOFError, ValueError):
    ppppp = PrettyProteinProjectPymolPluginPanel()
    # print('PrettyProteinProjectPymolPluginPanel FAILED TO LOAD FROM PICKLE')
    ppppp.init_session()

def main():
    print('RUNNING main if for debugging only!')

    while True:
        sleep(0.1)

if __name__ == '__main__':
    main()
