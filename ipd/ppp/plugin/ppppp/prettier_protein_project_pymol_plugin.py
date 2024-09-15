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
import traceback

_ppppp = None
SESSION_RESTORE = os.path.expanduser('~/.config/ppp/session_restore.pse')
PPPPP_PICKLE = os.path.expanduser('~/.config/ppp/PrettyProteinProjectPymolPluginPanel.pickle')
OPTS_PICKLE = os.path.expanduser('~/.config/ppp/PrettyProteinProjectPymolPluginPanel_opts.pickle')

_opts = ipd.Bunch(
    shuffle=0,
    hide_invalid=2,
    do_review_action=2,
    review_action='cp $file ./ppp/$poll_$grade_$filebase',
    prefetch=0,
    save_session=2,
    _strict=True,
    findcmd='',
    findpoll='',
)
_opts.cmds = [
    dict(
        name='Color By Chain',
        onstart=True,
        cmdon='util.cbc("elem C and subject")',
        cmdoff='util.cbag("subject")',
        public=True,
    ),
    dict(
        name='Show Ligand Interactions',
        onstart=False,
        cmdon='show sti, byres subject and elem N+O within 3 of het',
        cmdoff='hide sti, subject and not het',
        public=True,
    ),
]
_opts.private_polls = {
    'symgp c2 Ca binder':
    dict(name='symgp c2 Ca binder',
         desc='',
         path='/home/sheffler/project/rfdsym/abbas/pymol_saves',
         public=False,
         start=None,
         end=None)
}

def pipimport(pkgname):
    import importlib
    try:
        setattr(sys.modules[__name__], pkgname, importlib.import_module(pkgname))
    except ValueError:
        subprocess.check_call(f'{sys.executable} -mpip install requests {pkgname}'.split())

pipimport('requests')
pipimport('fuzzyfinder')

dtfmt = "%Y-%m-%dT%H:%M:%S.%f"

def isfalse_notify(ok, message):
    if not ok:
        pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)
        return True

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
        pymol.cmd.load(self.fname, 'subject')
        self.update()

    def update(self):
        for t in self.toggles.cmds:
            self.update_toggle(t)

    def update_toggle(self, toggle):
        if toggle: self.run_command(toggle.cmdon)
        else: self.run_command(toggle.cmdoff)

    def run_command(self, cmd):
        pymol.cmd.do(cmd)

class PPPClient:
    def __init__(self, server_addr):
        self.server_addr = server_addr

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
            if self.widget.checkState(): _opts.active_cmds.add(self.name.lower())
            else: _opts.active_cmds.remove(self.name.lower())
        if _ppppp.polls.activepoll and _ppppp.polls.activepoll.viewer:
            _ppppp.polls.activepoll.viewer.update_toggle(toggle=self)

    def __bool__(self):
        return bool(self.widget.checkState())

def copy_to_tempdir(tempdir, fname):
    basename = os.path.abspath(fname).replace('/', '\\')
    newname = os.path.join(tempdir, basename)
    return shutil.copyfile(fname, newname)

class FileFetcher(PartialPickleable):
    '''copy files to temp storage for fast access. use BackgroundFileFetcher unless it causes problems'''
    def __init__(self, fnames, seenit=None):
        self.fnames = fnames
        self.seenit = seenit or set()
        self.init_session()

    def init_session(self):
        if isinstance(self.fnames, (str, bytes)):
            if os.path.isdir(self.fnames):
                suffix = tuple('.pdb .pdb.gz .cif .bcif'.split())
                self.fnames = [os.path.abspath(_) for _ in os.listdir(self.fnames) if _.endswith(suffix)]
            else:
              with open(self.fnames) as inp:
                self.fnames = list(map(str.strip, inp))
        self.fnames = [_ for _ in self.fnames if _ not in self.seenit]
        if _opts.shuffle: self.pbdlist = random.shuffle(self.fnames)
        _ppppp.set_pbar(lb=0, val=len(self.seenit), ub=len(self.fnames) - 1)

    def new_file(self, fname):
        self.seenit.add(fname)
        _ppppp.set_pbar(lb=0, val=len(self.seenit), ub=len(self.fnames) - 1)
        return fname

    def __iter__(self):
        for fname in self.fnames:
            yield self.new_file(fname)

class BackgroundFileFetcher(FileFetcher):
    def __iter__(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with ThreadPoolExecutor(1) as exe:
                futures = exe.map(partial(copy_to_tempdir, tempdir), self.fnames)
                for fname in futures:
                    yield self.new_file(fname)

class PollInProgress():
    def __init__(self, poll):
        self.poll = poll
        self.viewer = None
        self.files = (BackgroundFileFetcher if _opts.prefetch else FileFetcher)(poll['path'])
        self.fileiter = iter(self.files)

    def init_session(self, client):
        self.client = client
        self.files.init_session()
        self.fileiter = iter(self.files)
        if self.viewer: self.viewer.init_session(_ppppp.toggles)

    def load_next(self):
        with contextlib.suppress(StopIteration):
            self.viewer = PymolFileViewer(next(self.fileiter), _ppppp.toggles)
            return True

    def record_review(self, grade, comment):
        if isfalse_notify(self.viewer, 'No working file!'): return
        print(f'REVIEWED: {self.viewer.fname}', flush=True)
        pymol.cmd.delete('subject')
        if not self.load_next():
            _ppppp.polls.poll_finished()

    def cleanup(self):
        pymol.cmd.delete('subject')

class Polls(PartialPickleable):
    def __init__(self):
        self.client = None
        self.activepoll = None

    def init_session(self, client, listwidget):
        self.client = client
        self.listwidget = listwidget
        self.listwidget.currentItemChanged.connect(lambda *a: self.poll_clicked(*a))
        self.newpollwidget = pymol.Qt.QtWidgets.QDialog()
        self.newpollwidget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'newpoll.ui'),
                                                   self.newpollwidget)
        self.newpollwidget.openfiledialog.clicked.connect(lambda: self.open_file_picker())
        self.newpollwidget.cancel.clicked.connect(lambda: self.newpollwidget.hide())
        self.newpollwidget.ok.clicked.connect(lambda: self.create_poll_done())
        self.refresh_polls()
        if self.activepoll:
            self.activepoll.init_session(self.client)

    def refresh_polls(self):
        self.public_polls = self.client.polls()
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
        polls = _opts.private_polls | self.public_polls
        if _opts.hide_invalid:
            polls = {n: p for n, p in polls.items() if self.pollstatus(p) != 'invalid'}
        if query := _opts.findpoll:
            namedesc = [f"{p['name'].lower()}||||{p['desc'].lower()}" for p in polls.values()]
            hits = {h[:h.find('||||')] for h in fuzzyfinder.fuzzyfinder(query.lower(), namedesc)}
            active = {}
            if self.activepoll: active[self.activepoll.poll['name']] = self.activepoll.poll
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
        for i, poll in enumerate(self.filtered_poll_list().values()):
            if isinstance(poll['start'], str):
                poll['start'] = datetime.datetime.strptime(poll['start'], dtfmt)
                poll['end'] = datetime.datetime.strptime(poll['end'], dtfmt)
            if poll['end'] and poll['end'] < datetime.datetime.now(): continue
            status = self.pollstatus(poll)
            self.listwidget.addItem(f"{poll['name']} ({status})")
            item = self.listwidget.item(i)
            item.setToolTip(poll['desc'])
            self.polls[i] = poll
            if self.activepoll and poll['pollid'] == self.activepoll.poll['pollid']:
                item.setSelected(True)

    def poll_clicked(self, new, old):
        assert self.listwidget.count() == len(self.polls)
        index = int(self.listwidget.indexFromItem(new).row())
        if index < 0:
            print('poll_clicked missing', new.text())
        if self.activepoll: self.activepoll.cleanup()
        self.activepoll = PollInProgress(self.polls[index])
        self.activepoll.load_next()

    def poll_finished(self):
        self.activepoll = None
        pymol.cmd.delete('subject')
        _ppppp.set_pbar(done=True)

    def create_poll_start(self):
        self.newpollwidget.show()

    def create_poll_done(self):
        self.newpollwidget.hide()
        duration = ipd.dev.safe_eval('dict(' + ','.join(self.newpollwidget.duration.text().split()) + ')')
        duration = datetime.timedelta(**duration)
        poll = dict(name=self.newpollwidget.name.text(),
                    desc=self.newpollwidget.desc.text(),
                    path=os.path.abspath(os.path.expanduser(self.newpollwidget.path.text())),
                    public=bool(self.newpollwidget.ispublic.checkState()),
                    telem=self.newpollwidget.telemetry.checkState(),
                    start=datetime.datetime.strftime(datetime.datetime.now(), dtfmt),
                    end=datetime.datetime.strftime(datetime.datetime.now() + duration, dtfmt))
        print('new poll', poll)
        self.client.create_poll(poll)
        if not poll['public']:
            poll['pollid'] = f'local{len(_opt.private_polls)}'
            _opt.private_polls[poll['name']] = poll
        self.refresh_polls()
        self.update_poll_list()

class ToggleCommands(PartialPickleable):
    def __init__(self):
        self.cmds = [ToggleCommand(**_) for _ in _opts.cmds]
        self.visible_cmds = self.cmds
        _opts.active_cmds = {cmd.name.lower() for cmd in self.cmds if cmd.onstart}

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
                   onstart=self.newopt.onstart.checkState(),
                   public=self.newopt.ispublic.checkState())
        self.cmds.append(ToggleCommand(**cmd))
        self.listwidget.addItem(self.cmds[-1].name)
        item = self.listwidget.item(self.listwidget.count() - 1)
        item.setCheckState(0)
        self.cmds[-1].init_session(item)
        if item.checkState():
            _opts.active_cmds.add(item.text().lower().replace(' (public)', '').replace(' (private)', ''))
        if self.cmds[-1].public:
            print('TODO: send new options to server')
        else:
            _opts.cmds.append(cmd)

    def update_toggle_list(self):
        print(_opts.active_cmds)
        cmds = self.cmds
        if query := _opts.findcmd:
            hits = set(fuzzyfinder.fuzzyfinder(query.lower(), [cmd.name.lower() for cmd in cmds]))
            hits |= _opts.active_cmds
            cmds = [cmd for cmd in self.cmds if cmd.name.lower() in hits]
        self.visible_cmds = cmds
        self.listwidget.clear()
        for cmd in cmds:
            self.listwidget.addItem(cmd.name)
            cmd.init_session(self.listwidget.item(self.listwidget.count() - 1))
            if cmd.name.lower() in _opts.active_cmds: cmd.widget.setCheckState(2)
            else: cmd.widget.setCheckState(0)

class PrettyProteinProjectPymolPluginPanel(PartialPickleable):
    def __init__(self, server_addr='localhost:12345'):
        self.server_addr = server_addr
        self.polls = Polls()
        self.toggles = ToggleCommands()

    def init_session(self):
        pymol.cmd.save(SESSION_RESTORE)
        self.client = PPPClient(self.server_addr)
        self.setup_main_window()
        self.setup_opts()
        self.polls.init_session(self.client, self.widget.polls)
        self.toggles.init_session(self.widget.toggles)

    def setup_main_window(self):
        self.widget = pymol.Qt.QtWidgets.QDialog()
        self.widget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'ppppp.ui'), self.widget)
        self.widget.show()
        for grade in 'SABCDF':
            getattr(self.widget, f'{grade.lower()}tier').clicked.connect(partial(self.record_review, grade))
        self.widget.button_newpoll.clicked.connect(lambda: self.polls.create_poll_start())
        self.widget.button_newopt.clicked.connect(lambda: self.toggles.create_toggle_start())
        self.widget.button_delopt.clicked.connect(lambda item: self.toggles.remove_toggle())
        self.widget.button_save_session.clicked.connect(lambda: self.save_session())
        self.widget.button_refresh.clicked.connect(lambda: self.polls.refresh_polls())
        self.widget.button_quit.clicked.connect(self.quit)
        pymol.cmd.set_key('left', lambda: self.polls.activepoll.prev())
        pymol.cmd.set_key('right', lambda: self.polls.activepoll.next())

    def setup_opts(self):
        action = collections.defaultdict(lambda: lambda: None)
        action['hide_invalid'] = self.polls.update_poll_list
        action['findpoll'] = self.polls.update_poll_list
        action['findcmd'] = self.toggles.update_toggle_list
        for name, widget in self.widget.__dict__.items():
            if not name.startswith('opt_'): continue
            opt = name[4:]
            if widget.__class__.__name__.endswith('QLineEdit'):
                widget.setText(_opts[opt])
                widget.textChanged.connect(lambda _, opt=opt: (setattr(_opts, opt, _), action[opt]()))
            elif widget.__class__.__name__.endswith('QCheckBox'):
                widget.setCheckState(_opts[opt])
                widget.stateChanged.connect(lambda _, opt=opt: (setattr(_opts, opt, _), action[opt]()))
            else:
                raise ValueError(f'dont know how to use option widget type {type(v)}')

    def record_review(self, grade):
        if isfalse_notify(self.polls.activepoll, 'No active poll!'): return
        comment = self.widget.comment.toPlainText()
        self.widget.comment.clear()
        self.polls.activepoll.record_review(grade, comment)

    def set_pbar(self, lb=None, val=None, ub=None, done=None):
        if done: return self.widget.progress.setProperty('enabled', False)
        self.widget.progress.setProperty('enabled', True)
        if lb is not None: self.widget.progress.setMinimum(lb)
        if ub is not None: self.widget.progress.setMaximum(ub)
        if val is not None: self.widget.progress.setValue(val)

    def save_session(self):
        if not _opts.save_session: return
        with open(OPTS_PICKLE, 'wb') as out:
            pickle.dump(_opts, out)
        with open(PPPPP_PICKLE, 'wb') as out:
            pickle.dump(self, out)
        print(f'PPP CONFIG SAVED, PollInProgress: {self.polls.activepoll}')

    def quit(self):
        self.save_session()
        self.widget.hide()
        if os.path.exists(SESSION_RESTORE):
            pymol.cmd.load(SESSION_RESTORE)
            os.remove(SESSION_RESTORE)

def run(_self=None):
    os.makedirs(os.path.dirname(SESSION_RESTORE), exist_ok=True)
    os.makedirs(os.path.dirname(PPPPP_PICKLE), exist_ok=True)
    if os.path.exists(SESSION_RESTORE): os.remove(SESSION_RESTORE)
    global _ppppp, _opts
    if os.path.exists(OPTS_PICKLE):
        with open(OPTS_PICKLE, 'rb') as inp:
            _opts |= pickle.load(inp)
    try:
        with open(PPPPP_PICKLE, 'rb') as inp:
            _ppppp = pickle.load(inp)
            if not _opts.save_session: raise ValueError
            print('PrettyProteinProjectPymolPluginPanel LOADED FROM PICKLE')
    except (FileNotFoundError, EOFError, ValueError):
        _ppppp = PrettyProteinProjectPymolPluginPanel('localhost:12345')
        print('PrettyProteinProjectPymolPluginPanel FAILED TO LOAD FROM PICKLE')
    _ppppp.init_session()
