import os
import sys
import subprocess
import pymol
import datetime
from functools import partial
import pickle
from icecream import ic
import traceback
import ipd

subprocess.check_call(f'{sys.executable} -mpip install requests fuzzyfinder'.split())
import requests
import fuzzyfinder

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
        return {p['name']: p for p in response.json()}

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

    def attach_widget(self, widget, init=False):
        self.widget = widget
        assert self.widget.text().startswith(self.name)
        self.widget.setText(f'{self.name} ({"Public" if self.public else "Private"})')
        self.widget.setFlags(self.widget.flags() | pymol.Qt.QtCore.Qt.ItemIsUserCheckable)
        # self.widget.setCheckState(2 * int(self.onstart))

    def widget_update(self, toggle=False):
        if toggle: self.widget.setCheckState(0 if self.widget.checkState() else 2)
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
        self.local_polls = {}
        self.active_poll = None
        self.query = None

    def __getstate__(self):
        return self.local_polls, self.active_poll, self.query

    def __setstate__(self, state):
        self.local_polls, self.active_poll, self.query = state

    def init_gui(self, client, listwidget):
        self.client = client
        self.public_polls = self.client.polls()
        self.listwidget = listwidget
        self.listwidget.currentItemChanged.connect(lambda *a: self.activate_poll(*a))
        self.newpollwidget = pymol.Qt.QtWidgets.QDialog()
        self.newpollwidget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'newpoll.ui'),
                                                   self.newpollwidget)
        self.newpollwidget.cancel.clicked.connect(lambda: self.newpollwidget.hide())
        self.newpollwidget.ok.clicked.connect(lambda: self.new_poll_done())
        self.update_poll_list()

    def filtered_poll_list(self, query):
        polls = self.local_polls | self.public_polls
        if query:
            hits = set(fuzzyfinder.fuzzyfinder(self.query.lower(), [name.lower() for name in polls]))
            active = {}
            if self.active_poll: active[self.active_poll.poll['name']] = self.active_poll.poll
            polls = active | {p['name']: p for p in polls.values() if p['name'].lower() in hits}
        return polls

    def update_poll_list(self, query=None):
        self.query = query
        self.listwidget.clear()
        self.polls = {}
        for i, poll in enumerate(self.filtered_poll_list(query).values()):
            if isinstance(poll['start'], str):
                poll['start'] = datetime.datetime.strptime(poll['start'], dtfmt)
                poll['end'] = datetime.datetime.strptime(poll['end'], dtfmt)
            if poll['end'] < datetime.datetime.now(): continue
            self.listwidget.addItem(f"{poll['name']} ({'Public' if poll['public'] else 'Private'})")
            item = self.listwidget.item(i)
            item.setToolTip(poll['desc'])
            self.polls[i] = poll
            if self.active_poll and poll['pollid'] == self.active_poll.poll['pollid']:
                item.setSelected(True)

    def activate_poll(self, new, old):
        assert self.listwidget.count() == len(self.polls)
        ic(new)
        ic(self.listwidget.indexFromItem(new))
        ic(self.listwidget.indexFromItem(new).row())
        index = int(self.listwidget.indexFromItem(new).row())
        self.active_poll = PollInProgress(self.polls[index])

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
            self.local_polls[poll['name']] = poll
        self.update_poll_list()

class ToggleListWidget:
    def __init__(self, cmds):
        self.cmds = cmds
        self.visible_cmds = cmds
        self.active_cmds = {cmd.name.lower() for cmd in cmds if cmd.onstart}
        self.query = None

    def __getstate__(self):
        return self.cmds, self.visible_cmds, self.active_cmds, self.query

    def __setstate__(self, state):
        self.cmds, self.visible_cmds, self.active_cmds, self.query = state

    def init_gui(self, listwidget):
        self.listwidget = listwidget
        self.update_toggle_list()
        # listwidget.itemChanged.connect(lambda _: self.update_item(_))
        self.listwidget.itemClicked.connect(lambda _: self.update_item(_, toggle=True))

    def apply_all(self):
        for cmd in self.cmds.values():
            cmd.apply()

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
        self.listwidget.item(self.listwidget.count() - 1)
        self.cmds[-1].attach_widget(item)
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
            cmd.attach_widget(self.listwidget.item(self.listwidget.count() - 1))
            if cmd.name.lower() in self.active_cmds: cmd.widget.setCheckState(2)
            else: cmd.widget.setCheckState(0)

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
        if self.polls.query: self.mainwidget.pollsearch.setText(self.polls.query)
        if self.toggles.query: self.mainwidget.cmdsearch.setText(self.toggles.query)

    def setup_main_window(self):
        self.mainwidget = pymol.Qt.QtWidgets.QDialog()
        self.mainwidget = pymol.Qt.utils.loadUi(os.path.join(os.path.dirname(__file__), 'ppppp.ui'),
                                                self.mainwidget)
        self.mainwidget.show()
        for grade in 'sabcf':
            getattr(self.mainwidget, f'{grade}tier').clicked.connect(partial(self.record_result, grade=grade))
        self.mainwidget.newpoll.clicked.connect(lambda: self.polls.new_poll_start())
        self.mainwidget.quit.clicked.connect(self.quit)
        self.mainwidget.newopt.clicked.connect(lambda: self.newopt.show())
        self.mainwidget.delopt.clicked.connect(lambda item: self.toggles.remove_toggle())
        self.mainwidget.saveconfig.clicked.connect(lambda: self.saveconfig())
        self.mainwidget.pollsearch.textChanged.connect(lambda query: self.polls.update_poll_list(query))
        self.mainwidget.cmdsearch.textChanged.connect(lambda query: self.toggles.update_toggle_list(query))

    def create_new_toggle(self):
        self.toggles.create_new(self.newopt)

    def record_result(self, grade, *args):
        assert not args
        if self.polls.active_poll is None:
            pymol.Qt.QtWidgets.QMessageBox.warning(None, "Warning", 'No active poll!')
            return
        comment = self.mainwidget.comment.toPlainText()
        self.mainwidget.comment.setText('Comment...')
        self.polls.active_poll.record_result(grade, comment)

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
    except (FileNotFoundError, EOFError, ValueError):
        print(traceback.format_exc())
        _ppppp = PPPPP('localhost:12345')
    _ppppp.init_gui()
