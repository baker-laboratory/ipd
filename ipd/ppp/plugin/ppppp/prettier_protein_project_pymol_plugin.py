import os
import sys
import subprocess
import datetime
from functools import partial
import random
import pickle
from _pickle import PicklingError
from icecream import ic
import traceback
import abc
import contextlib
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import ipd
import pymol

_ppppp = None
SESSION_RESTORE = os.path.expanduser('~/.config/ppp/session_restore.pse')
PPPPP_PICKLE = os.path.expanduser('~/.config/ppp/PrettyProteinProjectPymolPluginPanel.pickle')

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
        pymol.cmd.delete('not name fixture_*')
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
        if toggle: self.widget.setCheckState(0 if self.widget.checkState() else 2)
        if _ppppp.polls.pollinprogress:
            _ppppp.polls.pollinprogress.viewer.update_toggle(toggle=self)

    def __bool__(self):
        return bool(self.widget.checkState())

builtin_commands = [
    ToggleCommand(
        name='Color By Chain',
        onstart=True,
        cmdon='util.cbc("elem C and subject")',
        cmdoff='util.cbag("subject")',
        public=True,
    ),
    ToggleCommand(
        name='Show Ligand Interactions',
        onstart=False,
        cmdon='show sti, byres subject and elem N+O within 3 of het',
        cmdoff='hide sti, subject and not het',
        public=True,
    ),
]

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
            with open(self.fnames) as inp:
                self.fnames = list(map(str.strip, inp))
        self.fnames = [_ for _ in self.fnames if _ not in self.seenit]
        if _ppppp.opts.shuffle: self.pbdlist = random.shuffle(self.fnames)
        _ppppp.set_pbar(lb=0, val=len(self.seenit), ub=len(self.fnames) - 1)

    def new_file(self, fname):
        self.seenit.add(fname)
        _ppppp.set_pbar(lb=0, val=len(self.seenit), ub=len(self.fnames) - 1)
        return fname

    def __iter__(self):
        with tempfile.TemporaryDirectory() as tempdir:
            for fname in self.fnames:
                yield self.new_file(copy_to_tempdir(tempdir, fname))

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
        self.files = (BackgroundFileFetcher if _ppppp.opts.prefetch else FileFetcher)(poll['path'])
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
        if not self.load_next():
            _ppppp.polls.poll_finished()

class Polls(PartialPickleable):
    def __init__(self):
        self.client = None
        self.local_polls = {}
        self.pollinprogress = None
        self.query = None

    def init_session(self, client, listwidget):
        self.client = client
        self.listwidget = listwidget
        self.listwidget.currentItemChanged.connect(lambda *a: self.poll_clicked(*a))
        self.newpollwidget = pymol.Qt.QtWidgets.QDialog()
        self.newpollwidget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'newpoll.ui'),
                                                   self.newpollwidget)
        self.newpollwidget.cancel.clicked.connect(lambda: self.newpollwidget.hide())
        self.newpollwidget.ok.clicked.connect(lambda: self.create_poll_done())
        self.refresh_polls()
        if self.pollinprogress:
            self.pollinprogress.init_session(self.client)

    def refresh_polls(self):
        self.public_polls = self.client.polls()
        self.update_poll_list()

    def filtered_poll_list(self, query):
        polls = self.local_polls | self.public_polls
        if _ppppp.opts.hide_invalid:
            polls = {n: p for n, p in polls.items() if self.pollstatus(p) != 'invalid'}
        if query:
            namedesc = [f"{p['name']}||||{p['desc']}" for p in polls.values()]
            hits = {h[:h.find('||||')] for h in fuzzyfinder.fuzzyfinder(self.query.lower(), namedesc)}
            active = {}
            if self.pollinprogress: active[self.pollinprogress.poll['name']] = self.pollinprogress.poll
            polls = active | {p['name']: p for p in polls.values() if p['name'].lower() in hits}
        return polls

    def pollstatus(self, poll):
        if not os.path.exists(poll['path']): return 'invalid'
        if os.path.getsize(poll['path']) == 0: return 'invalid'
        if poll['public']: return 'public'
        return 'private'

    def update_poll_list(self, query=None):
        self.query = query or self.query
        self.listwidget.clear()
        self.polls = {}
        for i, poll in enumerate(self.filtered_poll_list(self.query).values()):
            if isinstance(poll['start'], str):
                poll['start'] = datetime.datetime.strptime(poll['start'], dtfmt)
                poll['end'] = datetime.datetime.strptime(poll['end'], dtfmt)
            if poll['end'] < datetime.datetime.now(): continue
            status = self.pollstatus(poll)
            self.listwidget.addItem(f"{poll['name']} ({status})")
            item = self.listwidget.item(i)
            item.setToolTip(poll['desc'])
            self.polls[i] = poll
            if self.pollinprogress and poll['pollid'] == self.pollinprogress.poll['pollid']:
                item.setSelected(True)

    def poll_clicked(self, new, old):
        assert self.listwidget.count() == len(self.polls)
        index = int(self.listwidget.indexFromItem(new).row())
        self.pollinprogress = PollInProgress(self.polls[index])
        self.pollinprogress.load_next()

    def poll_finished(self):
        self.pollinprogress = None
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
                    path=self.newpollwidget.path.text(),
                    public=bool(self.newpollwidget.ispublic.checkState()),
                    telem=self.newpollwidget.telemetry.checkState(),
                    start=datetime.datetime.strftime(datetime.datetime.now(), dtfmt),
                    end=datetime.datetime.strftime(datetime.datetime.now() + duration, dtfmt))
        print('new poll', poll)
        self.client.create_poll(poll)
        if not poll['public']:
            poll['pollid'] = f'local{len(self.local_polls)}'
            self.local_polls[poll['name']] = poll
        self.update_poll_list()

class ToggleListWidget(PartialPickleable):
    def __init__(self, cmds):
        self.cmds = cmds
        self.visible_cmds = cmds
        self.active_cmds = {cmd.name.lower() for cmd in cmds if cmd.onstart}
        self.query = None

    def init_session(self, listwidget):
        self.listwidget = listwidget
        self.update_toggle_list()
        # listwidget.itemChanged.connect(lambda _: self.update_item(_))
        self.listwidget.itemClicked.connect(lambda _: self.update_item(_, toggle=True))

    def update_item(self, item, toggle=False):
        assert len(self.visible_cmds) == self.listwidget.count()
        self.visible_cmds[self.listwidget.indexFromItem(item).row()].widget_update(toggle)
        if item.checkState():
            self.active_cmds.add(item.text().lower().replace(' (public)', '').replace(' (private)', ''))
        else:
            self.active_cmds.remove(item.text().lower().replace(' (public)', '').replace(' (private)', ''))

    def create_new(self, newopt):
        opt = dict(name=newopt.name.text(),
                   cmdon=newopt.cmdon.text(),
                   cmdoff=newopt.cmdoff.text(),
                   onstart=newopt.onstart.checkState(),
                   public=newopt.ispublic.checkState())
        self.cmds.append(ToggleCommand(**opt))
        self.listwidget.addItem(self.cmds[-1].name)
        item = self.listwidget.item(self.listwidget.count() - 1)
        item.setCheckState(0)
        self.cmds[-1].init_session(item)
        if item.checkState():
            self.active_cmds.add(item.text().lower().replace(' (public)', '').replace(' (private)', ''))
        if opt['public']:
            print('TODO: send new options to server')
        newopt.hide()

    def update_toggle_list(self, query=None):
        cmds = self.cmds
        if query:
            self.query = query
            hits = set(fuzzyfinder.fuzzyfinder(self.query, [cmd.name.lower() for cmd in cmds]))
            hits |= self.active_cmds
            cmds = [cmd for cmd in self.cmds if cmd.name.lower() in hits]
        self.visible_cmds = cmds
        self.listwidget.clear()
        for cmd in cmds:
            self.listwidget.addItem(cmd.name)
            cmd.init_session(self.listwidget.item(self.listwidget.count() - 1))
            if cmd.name.lower() in self.active_cmds: cmd.widget.setCheckState(2)
            else: cmd.widget.setCheckState(0)

class PrettyProteinProjectPymolPluginPanel(PartialPickleable):
    def __init__(self, server_addr='localhost:12345'):
        self.server_addr = server_addr
        self.polls = Polls()
        self.toggles = ToggleListWidget(builtin_commands)
        self.manual_state = {}
        self.opts = ipd.Bunch(shuffle=0, hide_invalid=2, do_on_rank=2, prefetch=0, save_session=2, _strict=True)

    def init_session(self):
        self.client = PPPClient(self.server_addr)
        self.setup_main_window()
        self.newopt = pymol.Qt.QtWidgets.QDialog()
        self.newopt = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'newopt.ui'), self.newopt)
        self.newopt.cancel.clicked.connect(lambda: self.newopt.hide())
        self.newopt.ok.clicked.connect(partial(self.toggles.create_new, newopt=self.newopt))
        self.polls.init_session(self.client, self.mainwidget.polls)
        self.toggles.init_session(self.mainwidget.toggles)
        if self.polls.query: self.mainwidget.pollsearch.setText(self.polls.query)
        if self.toggles.query: self.mainwidget.cmdsearch.setText(self.toggles.query)
        if 'comment' in self.manual_state: self.mainwidget.comment.setText(self.manual_state['comment'])
        self.setup_main_window_post_init()
        pymol.cmd.save(SESSION_RESTORE)

    def setup_main_window(self):
        self.mainwidget = pymol.Qt.QtWidgets.QDialog()
        self.mainwidget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'ppppp.ui'),
                                                self.mainwidget)
        self.mainwidget.show()
        for grade in 'SABCDF':
            getattr(self.mainwidget,
                    f'{grade.lower()}tier').clicked.connect(partial(self.record_review, grade=grade))
        pymol.cmd.set_key('left', self.record_review, 'D')
        pymol.cmd.set_key('right', self.record_review, 'A')
        self.mainwidget.button_newpoll.clicked.connect(lambda: self.polls.create_poll_start())
        self.mainwidget.button_quit.clicked.connect(self.quit)
        self.mainwidget.button_newopt.clicked.connect(lambda: self.newopt.show())
        self.mainwidget.button_delopt.clicked.connect(lambda item: self.toggles.remove_toggle())
        self.mainwidget.button_save_session.clicked.connect(lambda: self.save_session())
        self.mainwidget.button_refresh.clicked.connect(lambda: self.polls.refresh_polls())

    def setup_main_window_post_init(self):
        self.mainwidget.cmdsearch.textChanged.connect(lambda query: self.toggles.update_toggle_list(query))
        self.mainwidget.pollsearch.textChanged.connect(lambda query: self.polls.update_poll_list(query))
        self.mainwidget.opt_shuffle.stateChanged.connect(lambda val: setattr(self.opts, 'shuffle', val))
        self.mainwidget.opt_shuffle.setCheckState(self.opts.shuffle)
        self.mainwidget.opt_prefetch.stateChanged.connect(lambda val: setattr(self.opts, 'prefetch', val))
        self.mainwidget.opt_prefetch.setCheckState(self.opts.prefetch)
        self.mainwidget.opt_do_on_rank.stateChanged.connect(lambda val: setattr(self.opts, 'do_on_rank', val))
        self.mainwidget.opt_do_on_rank.setCheckState(self.opts.do_on_rank)
        self.mainwidget.opt_save_session.stateChanged.connect(lambda val: setattr(self.opts, 'save_session', val))
        self.mainwidget.opt_save_session.setCheckState(self.opts.save_session)
        self.mainwidget.opt_hide_invalid.stateChanged.connect(
            lambda val: (setattr(self.opts, 'hide_invalid', val), self.polls.update_poll_list()))
        self.mainwidget.opt_hide_invalid.setCheckState(self.opts.hide_invalid)

    def create_new_toggle(self):
        self.toggles.create_new(self.newopt)

    def record_review(self, grade):
        if isfalse_notify(self.polls.pollinprogress, 'No active poll!'): return
        comment = self.mainwidget.comment.toPlainText()
        self.mainwidget.comment.clear()
        self.polls.pollinprogress.record_review(grade, comment)

    def set_pbar(self, lb=None, val=None, ub=None, done=None):
        if done: return self.mainwidget.progress.setProperty('enabled', False)
        self.mainwidget.progress.setProperty('enabled', True)
        if lb is not None: self.mainwidget.progress.setMinimum(lb)
        if ub is not None: self.mainwidget.progress.setMaximum(ub)
        if val is not None: self.mainwidget.progress.setValue(val)

    def save_session(self):
        if not self.opts.save_session: return
        self.manual_state['comment'] = self.mainwidget.comment.toPlainText()
        with open(PPPPP_PICKLE, 'wb') as out: pickle.dump(self, out)
        print(f'CONFIG SAVED TO {PPPPP_PICKLE}, PollInProgress:', self.polls.pollinprogress)

    def quit(self):
        self.save_session()
        self.mainwidget.hide()
        if os.path.exists(SESSION_RESTORE):
            pymol.cmd.load(SESSION_RESTORE)
            os.remove(SESSION_RESTORE)

def run(_self=None):
    os.makedirs(os.path.dirname(SESSION_RESTORE), exist_ok=True)
    os.makedirs(os.path.dirname(PPPPP_PICKLE), exist_ok=True)
    if os.path.exists(SESSION_RESTORE): os.remove(SESSION_RESTORE)
    global _ppppp
    try:
        with open(PPPPP_PICKLE, 'rb') as inp:
            _ppppp = pickle.load(inp)
            if not _ppppp.opts.save_session: raise ValueError
            print('PrettyProteinProjectPymolPluginPanel LOADED FROM PICKLE')
    except (FileNotFoundError, EOFError, ValueError):
        _ppppp = PrettyProteinProjectPymolPluginPanel('localhost:12345')
        print('PrettyProteinProjectPymolPluginPanel FAILED TO LOAD FROM PICKLE')
    _ppppp.init_session()
