import os
import sys
import subprocess
import pymol
import datetime
from functools import partial
import pickle
import ipd
from icecream import ic

# subprocess.check_call(f'{sys.executable} -mpip install requests'.split())
import requests

dtfmt = "%Y-%m-%dT%H:%M:%S.%f"

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
        return response.json()

    def new_poll(self, poll):
        if not poll['public']: return
        self.post('/poll', json=poll).json()

class ToggleCommand:
    def __init__(self, name, onstart, cmdon, cmdoff, public=False):
        super(ToggleCommand, self).__init__()
        self.name = name
        self.onstart = onstart
        self.cmdon = cmdon
        self.cmdoff = cmdoff
        self.public = public
        self.widget = None

    def __getstate__(self):
        return (self.name, self.onstart, self.cmdon, self.cmdoff, self.public)

    def __setstate__(self, state):
        self.name, self.onstart, self.cmdon, self.cmdoff, self.public = state

    def attach_widget(self, widget):
        self.widget = widget
        assert self.widget.text().startswith(self.name)
        self.widget.setText(f'{self.name} ({"Public" if self.public else "Private"})')
        self.widget.setFlags(self.widget.flags() | pymol.Qt.QtCore.Qt.ItemIsUserCheckable)
        self.widget.setCheckState(pymol.Qt.QtCore.Qt.Checked if self.onstart else pymol.Qt.QtCore.Qt.Unchecked)

    def widget_update(self, toggle=False):
        if toggle: self.widget.setCheckState(not self.widget.checkState())
        pymol.cmd.do(self.cmdon if self.widget.checkState() else self.cmdoff)

    def apply(self):
        if self.widget.checkState():
            pymol.cmd.do(self.cmdon)

builtin_commands = [
    ToggleCommand(
        name='Color By Chain',
        onstart=True,
        cmdon='util.cbc("elem C")',
        cmdoff='util.cbag',
        public=True,
    ),
    ToggleCommand(
        name='Show Ligand Interactions',
        onstart=False,
        cmdon='show sti, byres elem N+O within 3 of het',
        cmdoff='hide sti',
        public=True,
    ),
]

class PollInProgress:
    def __init__(self, poll):
        self.poll = poll
        self.working_file = None
        print('new PollInProgress', poll)

    def record_result(self, grade, comment):
        if self.working_file is None:
            pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", 'No working file!')
            return
        print(grade, comment)
        if self.main.doaction.checkState():
            print(self.main.action)
class Polls:
    def __init__(self):
        self.client = None
        self.local_polls = []
        self.working_poll = None

    def __getstate__(self):
        return self.local_polls, self.working_poll

    def __setstate__(self, state):
        self.local_polls, self.working_poll = state

    def init_gui(self, client, listwidget):
        self.client = client
        self.listwidget = listwidget
        self.listwidget.currentItemChanged.connect(lambda *a: self.activate_poll(*a))
        self.newpollwidget = pymol.Qt.QtWidgets.QDialog()
        self.newpollwidget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'newpoll.ui'),
                                                   self.newpollwidget)
        self.newpollwidget.cancel.clicked.connect(lambda: self.newpollwidget.hide())
        self.newpollwidget.ok.clicked.connect(lambda: self.new_poll_done())
        self.update_poll_list()

    def update_poll_list(self):
        self.listwidget.clear()
        self.public_polls = self.client.polls()
        self.polls = {}
        for i, poll in enumerate(self.local_polls + self.public_polls):
            if isinstance(poll['start'], str):
                poll['start'] = datetime.datetime.strptime(poll['start'], dtfmt)
                poll['end'] = datetime.datetime.strptime(poll['end'], dtfmt)
            if poll['end'] < datetime.datetime.now(): continue
            self.listwidget.addItem(f"{poll['name']} ({'Public' if poll['public'] else 'Private'})")
            item = self.listwidget.item(i)
            item.setToolTip(poll['desc'])
            self.polls[i] = poll
            if self.working_poll and poll['pollid'] == self.working_poll.poll['pollid']:
                item.setSelected(True)

    def activate_poll(self, new, old):
        assert self.listwidget.count() == len(self.polls)
        ic(new)
        ic(self.listwidget.indexFromItem(new))
        ic(self.listwidget.indexFromItem(new).row())
        index = int(self.listwidget.indexFromItem(new).row())
        self.working_poll = PollInProgress(self.polls[index])

    def new_poll_start(self):
        self.newpollwidget.show()

    def new_poll_done(self):
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
        self.client.new_poll(poll)
        if not poll['public']:
            poll['pollid'] = f'local{len(self.local_polls)}'
            self.local_polls.append(poll)
        self.update_poll_list()

class ToggleListWidget:
    def __init__(self, cmds):
        self.cmds = cmds

    def __getstate__(self):
        return self.cmds

    def __setstate__(self, state):
        self.cmds = state

    def init_gui(self, widget):
        self.widget = widget
        for cmd in self.cmds:
            self.add_toggle(cmd)
        # widget.itemChanged.connect(lambda _: self.update_item(_))
        widget.itemClicked.connect(lambda _: self.update_item(_, toggle=True))

    def add_toggle(self, cmd):
        self.widget.addItem(cmd.name)
        cmd.attach_widget(self.widget.item(self.widget.count() - 1))

    def apply_all(self):
        for cmd in self.cmds.values():
            cmd.apply()

    def update_item(self, item, toggle=False):
        assert len(self.cmds) == self.widget.count()
        self.cmds[self.widget.indexFromItem(item).row()].widget_update(toggle)

    def create_new(self, newopt):
        opt = dict(name=newopt.name.text(),
                   cmdon=newopt.cmdon.text(),
                   cmdoff=newopt.cmdoff.text(),
                   onstart=newopt.onstart.checkState(),
                   public=newopt.ispublic.checkState())
        self.cmds.append(ToggleCommand(**opt))
        self.add_toggle(self.cmds[-1])
        if opt['public']:
            print('TODO: send new options to server')
        newopt.hide()

class PPPPP:
    def __init__(self, server_addr='localhost:12345'):
        self.server_addr = server_addr
        self.polls = Polls()
        self.toggles = ToggleListWidget(builtin_commands)

    def __getstate__(self):
        return self.server_addr, self.polls, self.toggles

    def __setstate__(self, state):
        self.server_addr, self.polls, self.toggles = state

    def init_gui(self):
        self.client = PPPClient(self.server_addr)
        self.setup_main_window()


        self.newopt = pymol.Qt.QtWidgets.QDialog()
        self.newopt = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'newopt.ui'), self.newopt)
        self.newopt.cancel.clicked.connect(lambda: self.newopt.hide())
        self.newopt.ok.clicked.connect(partial(self.toggles.create_new, newopt=self.newopt))
        self.polls.init_gui(self.client, self.mainwidget.polls)
        self.toggles.init_gui(self.mainwidget.toggles)

    def setup_main_window(self):
        self.mainwidget = pymol.Qt.QtWidgets.QDialog()
        self.mainwidget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'ppppp.ui'),
                                                self.mainwidget)
        for grade in 'sabcf':
            getattr(self.mainwidget, f'{grade}tier').clicked.connect(partial(self.record_result, grade=grade))
        self.mainwidget.newpoll.clicked.connect(lambda: self.polls.new_poll_start())
        self.mainwidget.quit.clicked.connect(self.quit)
        self.mainwidget.newopt.clicked.connect(lambda: self.newopt.show())
        self.mainwidget.saveconfig.clicked.connect(lambda: self.saveconfig())
        self.mainwidget.show()

    def create_new_toggle(self):
        self.toggles.create_new(self.newopt)

    def record_result(self, grade, *args):
        assert not args
        if self.polls.working_poll is None:
            pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", 'No active poll!')
            return
        comment = self.mainwidget.comment.toPlainText()
        self.mainwidget.comment.setText('Comment...')
        self.polls.working_poll.record_result(grade, comment)

    def saveconfig(self):
        configpath = os.path.expanduser('~/.config/ppp/ppppp.pickle')
        with open(configpath, 'wb') as out:
            pickle.dump(self, out)
        print(f'CONFIG SAVED TO {configpath}')

    def quit(self):
        self.saveconfig()
        self.mainwidget.hide()

_ppppp = None

def run(_self=None):
    global _ppppp
    if _ppppp: return
    configpath = os.path.expanduser('~/.config/ppp')
    os.makedirs(configpath, exist_ok=True)
    try:
        with open(configpath + '/ppppp.pickle', 'rb') as inp:
            _ppppp = pickle.load(inp)
            print('LOADED FROM PICKLE')
    except (FileNotFoundError, EOFError):
        _ppppp = PPPPP('localhost:12345')
    _ppppp.init_gui()
