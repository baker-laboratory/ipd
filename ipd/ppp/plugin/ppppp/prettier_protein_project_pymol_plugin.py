from _pickle import PicklingError
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from icecream import ic
import abc
import contextlib
import datetime
import collections
import ipd
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
import yaml

def pipimport(pkgname):
    import importlib
    try:
        setattr(sys.modules[__name__], pkgname, importlib.import_module(pkgname))
    except (ValueError, ModuleNotFoundError):
        subprocess.check_call(f'{sys.executable} -mpip install requests {pkgname}'.split())
        setattr(sys.modules[__name__], pkgname, importlib.import_module(pkgname))

pipimport('requests')
pipimport('fuzzyfinder')

remote, conf, ppppp = None, None, None
SERVER_ADDR = os.environ.get('PPPSERVER', 'localhost:12345')
CONFIG_DIR = os.path.expanduser('~/.config/ppp/')
CONFIG_FILE = f'{CONFIG_DIR}/localconfig.yaml'
STATE_FILE = f'{CONFIG_DIR}/localstate.yaml'
SESSION_RESTORE = f'{CONFIG_DIR}/session_restore.pse'
PPPPP_PICKLE = f'{CONFIG_DIR}/PrettyProteinProjectPymolPluginPanel.pickle'
SUFFIX = tuple('.pdb .pdb.gz .cif .bcif'.split())

def isfalse_notify(ok, message):
    if not ok:
        pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)
        return True

def havestate(name):
    if ppppp.polls.pollinprogress and name in state.polls[ppppp.polls.pollinprogress.name]: return True
    if name in state: return True
    if name in conf.opts: return True
    return False

def getstate(name, poll=None):
    defaults = dict(active_cmds=state.active_cmds,
                    review_action=conf.opts.review_action,
                    do_review_action=conf.opts.do_review_action,
                    findcmd=conf.opts.findcmd,
                    shuffle=conf.opts.shuffle,
                    prefetch=conf.opts.prefetch,
                    reviewed=set())
    if poll or ppppp.polls.pollinprogress:
        poll = poll or ppppp.polls.pollinprogress.name
        if name in state.polls[poll]:
            print('getstate', name, poll, state.polls[poll][name])
            return state.polls[poll][name]
        elif name in defaults:
            setattr(state.polls[poll], name, defaults[name])
            print('getstate', name, poll, state.polls[poll][name])
            return state.polls[poll][name]
    print('getstate', name, None)
    if name in state: return state[name]
    if name in conf.opts: return conf.opts[name]
    raise ValueError(f'getstate unknown state {name}')

def setstate(name, val, poll=None):
    getstate(name, poll)  # check already exists
    if poll or ppppp.polls.pollinprogress:
        poll = poll or ppppp.polls.pollinprogress.name
        print('setstate', name, val, poll)
        setattr(state[poll], name, val)
        return
    print('setstate', name, val, None)
    if name in state: state[name] = val
    if name in conf.opts: conf.opts[name] = val
    raise ValueError(f'setstate unknown state {name}')

_subject_count = 0

def subject_name():
    return 'subject%i' % _subject_count

def new_subject_name():
    global _subject_count
    _subject_count += 1
    return subject_name()

class PPPClient:
    def __init__(self, server_addr):
        self.server_addr = server_addr
        assert self.get('/').status_code == 200

    def get(self, url):
        return requests.get(f'http://{self.server_addr}/ppp{url}')

    def post(self, url, **kw):
        return requests.post(f'http://{self.server_addr}/ppp{url}', **kw)

    def polls(self):
        response = self.get('/poll')
        assert response.status_code == 200
        return {p['name']: p for p in response.json()}

    def create_poll(self, poll):
        if not poll['public']: return
        self.post('/poll', json=poll).json()

class PartialPickleable(abc.ABC):
    def __getstate__(self):
        state = {}
        for k, v in self.__dict__.items():
            state[k] = None
            with contextlib.suppress(TypeError, PicklingError):
                pickle.dumps(v)
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ |= state

    @abc.abstractmethod
    def init_session():
        pass

class PymolFileViewer(PartialPickleable):
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
        for cmd in getstate('active_cmds'):
            assert cmd in getstate('active_cmds')
            self.run_command(conf.cmds[cmd].cmdon)

    def update_toggle(self, toggle: 'ToggleCommand'):
        if toggle: self.run_command(toggle.cmdon)
        else: self.run_command(toggle.cmdoff)

    def run_command(self, cmd: str):
        pymol.cmd.do(cmd.replace('$subject', subject_name()))

    def cleanup(self):
        pymol.cmd.delete(subject_name())

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

class FileCache:
    '''
    Copies files to a conf temp directory. Will downloads files ahead of requested index in background.
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
        return f"{self.path}/{fname.replace('/','\\')}"

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
        fnames = [os.path.join(fnames, _) for _ in os.listdir(fnames) if _.endswith(SUFFIX)]
    else:
        with open(fnames) as inp:
            fnames = list(map(os.path.abspath, map(str.strip, inp)))
        if not all(f.endswith(SUFFIX) or os.path.isdir(f) for f in fnames): return None
        if not all(os.path.exists(f) for f in fnames): return None
    return fnames

class PollInProgress():
    def __init__(self, poll):
        self.poll = poll
        self.name = poll['name']
        self.viewer = None
        self.filecache = FileCache(self.fnames, numprefetch=7 if conf.prefetch else 0)
        ppppp.set_pbar(0, len(getstate('reviewed', poll=self.name)), len(self.fnames))

    def init_files(self, fnames):
        if isinstance(fnames, (str, bytes)):
            fnames = fnames_from_path(fnames)
        # fnames = [_ for _ in fnames if _ not in state.polls.reviewed]
        if conf.opts.shuffle: self.pbdlist = random.shuffle(fnames)
        ppppp.set_pbar(lb=0, val=len(getstate('reviewed')), ub=len(fnames) - 1)
        return fnames

    def init_session(self):
        if self.viewer: self.viewer.init_session(ppppp.toggles)

    @property
    def fnames(self):
        if 'fnames' not in state.polls[self.name]:
            state.polls[self.name].fnames = self.init_files(self.poll['path'])
        return state.polls[self.name].fnames

    @property
    def index(self):
        if 'activepollindex' not in state.polls[self.name]:
            state.polls[self.name].activepollindex = 0
        return state.polls[self.name].activepollindex

    @index.setter
    def index(self, index):
        state.polls[self.name].activepollindex = index

    def start(self):
        ppppp.update_opts()
        self.switch_to(self.index)

    def load_next(self):
        return self.switch_to(self.index + 1)

    def switch_to(self, index=None, delta=None):
        if index is None: index = self.index
        if delta: index += delta
        if index >= len(self.fnames): return False
        self.viewer = PymolFileViewer(self.filecache[index], ppppp.toggles)
        self.index = index
        return True

    def record_review(self, grade, comment):
        if isfalse_notify(self.viewer, 'No working file!'): return
        print(f'REVIEWED: {self.viewer.fname}', flush=True)
        pymol.cmd.delete(subject_name())
        print(state.polls[self.name])
        getstate('reviewed').add(self.viewer.fname)
        ppppp.set_pbar(lb=0, val=len(getstate('reviewed')), ub=len(self.fnames) - 1)
        if not self.load_next(): ppppp.polls.poll_finished()

    def cleanup(self):
        if self.viewer: self.viewer.cleanup()
        self.filecache.cleanup()

class Polls(PartialPickleable):
    def __init__(self):
        self.pollinprogress = None
        self.current_poll_index = None

    def init_session(self, listwidget):
        self.listwidget = listwidget
        self.listwidget.itemClicked.connect(lambda a: self.poll_clicked(a))
        self.newpollwidget = pymol.Qt.QtWidgets.QDialog()
        self.newpollwidget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'newpoll.ui'),
                                                   self.newpollwidget)
        self.newpollwidget.openfiledialog.clicked.connect(lambda: self.open_file_picker())
        self.newpollwidget.cancel.clicked.connect(lambda: self.newpollwidget.hide())
        self.newpollwidget.ok.clicked.connect(lambda: self.create_poll_done())
        self.refresh_polls()
        if state.activepoll:
            if matches := [i for i, x in self.polls.items() if x['name'] == state.activepoll]:
                self.poll_start(matches[0])

    def refresh_polls(self):
        self.public_polls = remote.polls()
        self.update_poll_list()

    def open_file_picker(self):
        dialog = pymol.Qt.QtWidgets.QFileDialog(self.newpollwidget)
        dialog.setFileMode(pymol.Qt.QtWidgets.QFileDialog.Directory)
        dialog.setDirectory(os.path.expanduser('~'))
        dialog.show()
        if dialog.exec_():
            file_names = dialog.selectedFiles()
            assert len(file_names) == 1
            self.newpollwidget.path.setText(file_names[0])

    def filtered_poll_list(self):
        # conf._autoreload_check()
        polls = conf.polls | self.public_polls
        if conf.opts.hide_invalid:
            polls = {n: p for n, p in polls.items() if self.pollstatus(p) != 'invalid'}
        if query := conf.opts.findpoll:
            namedesc = [f"{p['name'].lower()}||||{p['desc'].lower()}" for p in polls.values()]
            hits = {h[:h.find('||||')] for h in fuzzyfinder.fuzzyfinder(query.lower(), namedesc)}
            active = {}
            if self.pollinprogress: active[self.pollinprogress.name] = self.pollinprogress.poll
            polls = active | {p['name']: p for p in polls.values() if p['name'].lower() in hits}
        return polls

    def pollstatus(self, poll):
        if not os.path.exists(poll['path']): return 'invalid'
        if os.path.getsize(poll['path']) == 0: return 'invalid'
        if poll['public']: return 'public'
        return 'private'

    def update_poll_list(self):
        self.listwidget.clear()
        self.polls = {}
        filteredpolls = self.filtered_poll_list()
        for i, poll in enumerate(filteredpolls.values()):
            if isinstance(poll['start'], str):
                poll['start'] = datetime.datetime.strptime(poll['start'], '%Y-%m-%dT%H:%M:%S.%f')
                poll['end'] = datetime.datetime.strptime(poll['end'], '%Y-%m-%dT%H:%M:%S.%f')
            if poll['end'] and poll['end'] < datetime.datetime.now(): continue
            self.polls[i] = poll
            status = self.pollstatus(poll)
            self.listwidget.addItem(f"{poll['name']} ({status})")
            item = self.listwidget.item(i)
            item.setToolTip(poll['desc'])
            if state.activepoll and poll['pollid'] == filteredpolls[state.activepoll]['pollid']:
                item.setSelected(True)

    def poll_clicked(self, item):
        assert self.listwidget.count() == len(self.polls)
        assert item.isSelected()
        if item.isSelected():
            index = int(self.listwidget.indexFromItem(item).row())
            if index < 0: print('poll selected item missing', item.text())
            if self.current_poll_index == index: return
            # print('poll selected', item.text())
            self.poll_start(index)
        else:
            print('poll finished', item.text())
            self.poll_finished()

    def poll_start(self, index):
        if not 0 <= index < len(self.polls): print('bad poll index', index)
        if self.pollinprogress: self.pollinprogress.cleanup()
        self.pollinprogress = PollInProgress(self.polls[index])
        state.activepoll = self.pollinprogress.name
        self.current_poll_index = index
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

    def create_poll_done(self):
        duration = ipd.dev.safe_eval('dict(' + ','.join(self.newpollwidget.duration.text().split()) + ')')
        duration = datetime.timedelta(**duration)
        duration = duration or datetime.timedelta(weeks=99999)
        if isfalse_notify(self.newpollwidget.name.text(), 'Must provide a Name'): return
        if isfalse_notify(self.newpollwidget.path.text(), 'Must provide a Path'): return
        if isfalse_notify(duration > datetime.timedelta(minutes=1), 'Poll expires too soon'): return
        poll = dict(name=self.newpollwidget.name.text(),
                    desc=self.newpollwidget.desc.text(),
                    path=self.newpollwidget.path.text(),
                    public=2 if self.newpollwidget.ispublic.checkState() else 0,
                    telem=2 if self.newpollwidget.telemetry.checkState() else 0,
                    start=datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H:%M:%S.%f'),
                    end=datetime.datetime.strftime(datetime.datetime.now() + duration, '%Y-%m-%dT%H:%M:%S.%f'))
        if self.create_poll(poll):
            self.newpollwidget.hide()

    def create_poll(self, poll, temporary=False):
        poll['path'] = os.path.abspath(os.path.expanduser(poll['path']))
        if isfalse_notify(os.path.exists(poll['path']), 'path does not exist!'): return
        if isfalse_notify(fnames_from_path(poll['path']), f'contains no files with suffix {SUFFIX}'): return
        if poll['public']:
            remote.create_poll(poll)
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
            poll['pollid'] = f'conf{len(conf.polls)}'
            setattr(conf.polls, poll['name'], poll)
            self.refresh_polls()
        self.update_poll_list()
        return True

    def create_poll_from_curdir(self):
        self.create_poll(dict(name='Files in curent directory',
                              desc='The poll for lazy people',
                              path='.',
                              public=False,
                              telem=False,
                              start=None,
                              end=None,
                              temporary=True),
                         temporary=True)

class ToggleCommand(PartialPickleable):
    def __init__(self, name, onstart, cmdon, cmdoff, public=False):
        super(ToggleCommand, self).__init__()
        self.name = name
        self.onstart = onstart
        self.cmdon = cmdon
        self.cmdoff = cmdoff
        self.public = public
        self.status = 'public' if public else 'private'
        self.widget = None

    def init_session(self, widget):
        self.widget = widget
        assert self.widget.text().startswith(self.name)
        self.widget.setText(f'{self.name} ({self.status})')
        self.widget.setFlags(self.widget.flags() | pymol.Qt.QtCore.Qt.ItemIsUserCheckable)
        # self.widget.setCheckState(2 * int(self.onstart))

    def widget_update(self, toggle=False):
        if toggle:
            self.widget.setCheckState(0 if self.widget.checkState() else 2)
            if self.widget.checkState(): getstate('active_cmds').add(self.name)
            else: getstate('active_cmds').remove(self.name)
        if ppppp.polls.pollinprogress and ppppp.polls.pollinprogress.viewer:
            print('toggle update')
            ppppp.polls.pollinprogress.viewer.update_toggle(toggle=self)

    def __bool__(self):
        return bool(self.widget.checkState())

def copy_to_tempdir(tempdir, fname):
    basename = os.path.abspath(fname).replace('/', '\\')
    newname = os.path.join(tempdir, basename)
    return shutil.copyfile(fname, newname)

class ToggleCommands(PartialPickleable):
    def __init__(self):
        self.cmds = [ToggleCommand(**_) for _ in conf.cmds.values()]
        self.visible_cmds = self.cmds
        if not getstate('active_cmds'):
            setattr(conf.opts, 'active_cmds', {cmd.name for cmd in self.cmds if cmd.onstart})
            assert type(getstate('active_cmds')) != set

    def init_session(self, listwidget):
        self.listwidget = listwidget
        self.update_toggle_list()
        # listwidget.itemChanged.connect(lambda _: self.update_item(_))
        self.listwidget.itemClicked.connect(lambda _: self.update_item(_, toggle=True))
        self.newopt = pymol.Qt.QtWidgets.QDialog()
        self.newopt = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'newopt.ui'), self.newopt)
        self.newopt.cancel.clicked.connect(lambda: self.newopt.hide())
        self.newopt.ok.clicked.connect(lambda: self.create_toggle_done())

    def update_item(self, item, toggle=False):
        assert len(self.visible_cmds) == self.listwidget.count()
        self.visible_cmds[self.listwidget.indexFromItem(item).row()].widget_update(toggle)
        self.update_toggle_list()

    def create_toggle_start(self):
        self.newopt.show()

    def create_toggle_done(self):
        self.newopt.hide()
        cmd = dict(name=self.newopt.name.text(),
                   cmdon=self.newopt.cmdon.text(),
                   cmdoff=self.newopt.cmdoff.text(),
                   onstart=2 if self.newopt.onstart.checkState() else 0,
                   public=2 if self.newopt.ispublic.checkState() else 0)
        self.cmds.append(ToggleCommand(**cmd))
        self.listwidget.addItem(self.cmds[-1].name)
        item = self.listwidget.item(self.listwidget.count() - 1)
        item.setCheckState(0)
        self.cmds[-1].init_session(item)
        if item.checkState():
            getstate('active_cmds').add(item.text().lower().replace(' (public)', '').replace(' (private)', ''))
        if self.cmds[-1].public:
            print('TODO: send new options to server')
        else:
            setattr(conf.cmds, cmd['name'], cmd)

    def update_toggle_list(self):
        cmds = self.cmds
        if conf.opts.findcmd:
            hits = set(fuzzyfinder.fuzzyfinder(conf.opts.findcmd.lower(), [cmd.name.lower() for cmd in cmds]))
            hits |= {_.lower() for _ in getstate('active_cmds')}
            cmds = [cmd for cmd in self.cmds if cmd.name.lower() in hits]
        self.visible_cmds = cmds
        self.listwidget.clear()
        for cmd in cmds:
            self.listwidget.addItem(cmd.name)
            item = self.listwidget.item(self.listwidget.count() - 1)
            item.setToolTip(f'NAME: {cmd.name}\nON: {cmd.cmdon}\nOFF: {cmd.cmdoff}\nPUBLIC: {cmd.public}')
            cmd.init_session(item)
            cmd.widget.setCheckState(2) if cmd.name in getstate('active_cmds') else cmd.widget.setCheckState(0)

class PrettyProteinProjectPymolPluginPanel(PartialPickleable):
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
        self.widget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'ppppp.ui'), self.widget)
        self.widget.show()
        for grade in 'SABCDF':
            getattr(self.widget, f'{grade.lower()}tier').clicked.connect(partial(self.record_review, grade))
        self.widget.button_newpoll.clicked.connect(lambda: self.polls.create_poll_start())
        self.widget.button_use_curdir.clicked.connect(lambda: self.polls.create_poll_from_curdir())
        self.widget.button_newopt.clicked.connect(lambda: self.toggles.create_toggle_start())
        self.widget.button_save.clicked.connect(lambda: self.save_session())
        self.widget.button_load.clicked.connect(lambda: self.load_session())
        self.widget.button_restart.clicked.connect(lambda: self.init_session())
        self.widget.button_quit.clicked.connect(self.quit)
        self.keybinds = []
        self.add_keybind('left', 'LeftArrow', lambda: self.polls.pollinprogress.switch_to(delta=-1))
        self.add_keybind('right', 'RightArrow', lambda: self.polls.pollinprogress.switch_to(delta=1))

    def add_keybind(self, key, qkey, action):
        if qkey:
            keybind = pymol.Qt.QtWidgets.QShortcut(qkey, self.widget)
            keybind.activated.connect(action)
            self.keybinds.append(keybind)
        pymol.cmd.set_key(key, action)

    def update_opts(self):
        print('UPDATE OPTS', ppppp.polls.pollinprogress)
        action = collections.defaultdict(lambda: lambda: None)
        action['hide_invalid'] = self.polls.update_poll_list
        action['findpoll'] = self.polls.update_poll_list
        action['findcmd'] = self.toggles.update_toggle_list
        for name, widget in self.widget.__dict__.items():
            if not name.startswith('opt_'): continue
            opt = name[4:]
            if widget.__class__.__name__.endswith('QLineEdit'):
                if havestate(opt): widget.setText(getstate(opt))
                else: setstate(opt, widget.text())
                widget.textChanged.connect(lambda _, opt=opt: (setstate(opt, _), action[opt]()))
            elif widget.__class__.__name__.endswith('QCheckBox'):
                if havestate(opt): widget.setCheckState(getstate(opt))
                else: setstate(opt, 2 if widget.checkState() else 0)
                widget.stateChanged.connect(lambda _, opt=opt: (setstate(opt, _), action[opt]()))
            else:
                raise ValueError(f'dont know how to use option widget type {type(v)}')

    def record_review(self, grade):
        if isfalse_notify(self.polls.pollinprogress, 'No active poll!'): return
        comment = self.widget.comment.toPlainText()
        self.widget.comment.clear()
        self.polls.pollinprogress.record_review(grade, comment)

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

    def quit(self):
        self.save_session()
        self.widget.hide()
        if os.path.exists(SESSION_RESTORE):
            pymol.cmd.load(SESSION_RESTORE)
            os.remove(SESSION_RESTORE)

def run_local_server(port=12345):
    pipimport('fastapi')
    pipimport('sqlmodel')
    pipimport('uvicorn')
    args = dict(port=port, log='warning', datadir=os.path.expanduser('~/.config/localserver'))
    __server_thread = threading.Thread(target=ipd.ppp.server.run, kwargs=args, daemon=True)
    __server_thread.start()
    # dialog should cover server start time
    isfalse_notify(False, f"Can't connt to: {SERVER_ADDR}\nWill try to run a conf server.")
    return PPPClient(f'localhost:{port}')

def read_config(fname, **kw):
    result = ipd.dev.Bunch(**kw)
    if os.path.exists(fname):
        with open(fname) as inp:
            result |= ipd.Bunch(yaml.load(inp, yaml.CLoader))
    return ipd.dev.make_autosave_hierarchy(result, _autosave=fname, _default='bunchwithparent')

def run(_self=None):
    os.makedirs(os.path.dirname(SESSION_RESTORE), exist_ok=True)
    os.makedirs(os.path.dirname(PPPPP_PICKLE), exist_ok=True)
    if os.path.exists(SESSION_RESTORE): os.remove(SESSION_RESTORE)

    global ppppp, conf, state, remote
    conf = read_config(CONFIG_FILE, opts={}, cmds={}, polls={})
    state = read_config(STATE_FILE, activepoll=None, active_cmds=set(), cmds={}, polls={})
    for k in [k for k, v in conf.polls.items() if v.temporary]:
        del conf.polls[k]
    try:
        remote = PPPClient(SERVER_ADDR)
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
