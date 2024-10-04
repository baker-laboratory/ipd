from functools import partial
from icecream import ic
import datetime
import collections
import ipd
from ipd import ppp
import os
import pymol
import random
import getpass
import subprocess
from subprocess import check_output
import sys
import pydantic
import time
import traceback
from rich import print
from ipd.qt import MenuAction as CTXM, notify, isfalse_notify

it = ipd.lazyimport('itertools', 'more_itertools', pip=True)
requests = ipd.lazyimport('requests', pip=True)
# fuzzyfinder = ipd.lazyimport('fuzzyfinder', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
wu = ipd.lazyimport('willutil', 'git+https://github.com/willsheffler/willutil.git', pip=True)
wpc = ipd.lazyimport('wills_pymol_crap', 'git+https://github.com/willsheffler/wills_pymol_crap.git', pip=True)
wu.h
wpc.pymol_util

remote, state, ppppp = None, None, None
ISGLOBALSTATE, ISPERPOLLSTATE = set(), set()
CONFIG_DIR = os.path.expanduser('~/.config/ppp/')
CONFIG_FILE = f'{CONFIG_DIR}/localconfig.yaml'
STATE_FILE = f'{CONFIG_DIR}/localstate.yaml'
SESSION_RESTORE = f'{CONFIG_DIR}/session_restore.pse'
PPPPP_PICKLE = f'{CONFIG_DIR}/PrettyProteinProjectPymolPluginPanel.pickle'
TEST_STATE = {}

# profile = ipd.dev.timed
profile = lambda f: f

def ppp_pymol_add_default(name, val, isglobal=False):
    name = f'ppp_pymol_{name}'
    DEFAULTS[name] = val
    if state: state._statetype[name] = 'global' if isglobal else 'perpoll'

def ppp_pymol_get(name):
    if not ppppp: return TEST_STATE[name]
    # print('ppp get', state[f'ppp_pymol_{name}'])
    return state[f'ppp_pymol_{name}']

def ppp_pymol_set(name, val):
    # print('PYMOLSET', name, val, ppppp)
    if not ppppp: TEST_STATE[name] = val
    else: state[f'ppp_pymol_{name}'] = val

def widget_gettext(widget):
    if hasattr(widget, 'text'): return widget.text()
    if hasattr(widget, 'toPlainText'): return widget.toPlainText()
    if hasattr(widget, 'currentText'): return widget.currentText()

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
        if state.pymol_view: pymol.cmd.set_view(state.pymol_view)

    def update_toggle(self, toggle: 'ToggleCommand'):
        if toggle: self.run_command(toggle.cmdon)
        else: self.run_command(toggle.cmdoff)

    def run_command(self, cmd: str):
        assert isinstance(cmd, str)
        pymol.cmd.do(cmd.replace('$subject', subject_name()))

    def cleanup(self):
        pymol.cmd.delete(subject_name())
        state.pymol_view = pymol.cmd.get_view()

class PollInProgress:
    def __init__(self, poll):
        self.poll = poll
        self.viewer = None
        state.activepoll = poll.name
        Cache = ipd.dev.PrefetchLocalFileCache if state.prefetch else ipd.devFileCache
        self.filecache = Cache(self.fnames, numprefetch=7 if state.prefetch else 0)
        ppppp.toggles.update_toggles_gui()
        ppppp.set_pbar(lb=0, val=len(state.reviewed), ub=len(self.fnames) - 1)

    def init_files(self):
        fnames = [f.fname for f in self.poll.files]
        if state.shuffle: self.pbdlist = random.shuffle(fnames)
        return fnames

    @property
    def fnames(self):
        if not state.fnames: state.fnames = self.init_files()
        return state.fnames

    @property
    def index(self):
        if not state.activepollindex: state.activepollindex = 0
        return state.activepollindex

    @index.setter
    def index(self, index):
        state.activepollindex = index

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
        response = remote.upload_review(review, self.fnames[self.index])
        if isfalse_notify(not response, f'upload file server response: {response}'): return
        self.review_accepted(review)

    def review_accepted(self, review):
        pymol.cmd.delete(subject_name())
        state.reviewed.add(self.viewer.fname)
        ppppp.set_pbar(lb=0, val=len(state.reviewed), ub=len(self.fnames) - 1)
        ppppp.widget.comment.setText('')
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
        if review.grade in ('dislike', 'hate'): return True
        cmds = self.preprocess_shell_cmd(state.review_action.replace('$grade', review.grade))
        for cmd in cmds:
            try:
                result = check_output(cmd, stderr=subprocess.STDOUT)
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

class Polls(ipd.qt.ContextMenuMixin):
    def __init__(self, state):
        self.state = state
        self.pollinprogress = None
        self.current_poll_index = None
        self.listitems = None
        self.Qt = pymol.Qt

    def _context_menu_items(self):
        shufstr = "shuffle {'off' if state.polls[item.text()].shuffle else 'on'}"
        return {
            'details': CTXM(func=lambda cmd: ipd.qt.notify(cmd)),
            'refersh': CTXM(func=self.refresh_polls, item=False),
            'edit': CTXM(func=self.edit_poll, owner=True),
            'delete': CTXM(func=self.delete_poll, owner=True),
            'shuffle on/off': CTXM(func=self.shuffle_poll, item=True),
            'prefetch on/off': CTXM(func=self.prefetch_poll, item=True),
        }

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

    def get_from_item(self, item):
        return remote.poll(self.allpolls[item.text()])

    def shuffle_poll(self, poll):
        if poll: state.polls[poll.name].shuffle = not state.polls[poll.name].shuffle
        else: state._conf.shuffle = not state._conf.shuffle

    def prefetch_poll(self, poll):
        if poll: state.polls[poll.name].prefetch = not state.polls[poll.name].prefetch
        else: state._conf.prefetch = not state._conf.prefetch

    def edit_poll(self, poll):
        print('context poll edit', poll.name)

    def delete_poll(self, poll):
        if state.activepoll == poll.name:
            state.activepoll = None
        remote.remove(poll)
        self.refresh_polls()

    def refresh_polls(self):
        # localpolls = [(p.dbkey, p.name, p.user, p.desc, p.sym, p.ligand) for p in state.local.polls.values()]
        self.pollsearchtext, self.polltooltip, allpolls = [], {}, {}
        self.listitems, self.listitemdict = [], {}
        self.allpolls = remote.pollinfo(user=state.user)  #+ localpolls
        if not self.allpolls: return
        for key, name, user, desc, sym, lig, nchain in self.allpolls:
            ttip = f'NAME: {name}\nDESCRIPTION: DBKEY:{key}\n{desc}\nSYM: {sym}\nUSER: {user}\nLIG: {lig}\nNCHAIN: {nchain}'
            self.polltooltip[name] = ttip
            self.pollsearchtext.append(f'{name}||||{desc} sym:{sym} user:{user} lig:{lig}')
            allpolls[name] = key
        self.allpolls = allpolls
        self.pollsearchtext = '\n'.join(self.pollsearchtext)
        self.widget.clear()
        if self.widget.count() == 0:
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
        self.newpollwidget.user.setText(state.user)
        self.newpollwidget.show()

    def create_poll_spec_from_gui(self):
        # print('create_poll_spec_from_gui')
        # sourcery skip: dict-assign-update-to-union
        duration = ipd.dev.safe_eval('dict(' + ','.join(self.newpollwidget.duration.text().split()) + ')')
        duration = datetime.timedelta(**duration)
        duration = duration or datetime.timedelta(weeks=99999)
        # if isfalse_notify(self.newpollwidget.name.text(), 'Must provide a Name'): return
        # if isfalse_notify(os.path.exists(os.path.expanduser(self.newpollwidget.path.text())),
        # 'path must exist'):
        # return
        if isfalse_notify(duration > datetime.timedelta(minutes=1), 'Poll expires too soon'): return
        fields = 'name path sym ligand user workflow cmdstart cmdstop props attrs'
        kw = {k: widget_gettext(getattr(self.newpollwidget, k)) for k in fields.split()}
        kw |= {k: bool(getattr(self.newpollwidget, k).checkState()) for k in 'ispublic telemetry'.split()}
        kw['enddate'] = datetime.datetime.now() + duration
        return self.create_poll_spec(**kw)

    def create_poll_autofill_button(self):
        # print('create_poll_autofill_button')
        if poll := self.create_poll_spec_from_gui():
            for k in 'name path sym ligand user nchain'.split():
                getattr(self.newpollwidget, k).setText(str(poll[k]))

    def create_poll_ok_button(self):
        poll = self.create_poll_spec_from_gui()
        if self.create_poll(poll):
            self.newpollwidget.hide()

    def create_poll(self, poll: ppp.PollSpec, temporary=False):
        if not poll: return False
        response = remote.upload_poll(poll)
        if isfalse_notify(not response, f'server response: {response}'): return False
        self.refresh_polls()
        return True

    def create_poll_spec(self, **kw):
        # print('create_poll_spec')
        try:
            return ppp.PollSpec(**kw)
        except Exception as e:
            notify(f'create PollSpec error:\n{e}\n{traceback.format_exc()}')
            return None

    def create_poll_from_curdir(self):
        u = state.user
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
            # print('toggle update')
            ppppp.polls.pollinprogress.viewer.update_toggle(toggle=self)

    def __bool__(self):
        return bool(self.widget.checkState())

class ToggleCommands(ipd.qt.ContextMenuMixin):
    def __init__(self, state):
        self.state = state
        self.Qt = pymol.Qt
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

    def _context_menu_items(self):
        return {
            'details': CTXM(func=lambda cmd: ipd.qt.notify(cmd)),
            'refersh': CTXM(func=self.refersh_toggle_list, item=False),
            'edit': CTXM(func=self.edit_toggle, owner=True),
            'delete': CTXM(func=self.delete_toggle, owner=True),
        }

    def get_from_item(self, item):
        return self.cmds[item.text()]

    def edit_toggle(self, toggle):
        print('context toggle edit', toggle.name)

    def delete_toggle(self, toggle):
        remote.remove(toggle)
        self.refersh_toggle_list()

    def update_item(self, item, toggle=False):
        self.cmds[item.text()].widget_update(toggle)

    def create_toggle_start(self):
        self.gui_new_pymolcmd.user.setText(state.user)
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
            cmd = ppp.PymolCMD(None, dbkey=len(state.cmds) + 1, **cmdspec.dict())
            setattr(state.cmds, cmd.name, cmd)
        self.refersh_toggle_list()
        self.update_toggles_gui()

    def refersh_toggle_list(self):
        assert self.widget is not None
        cmdsdicts = list(state.cmds.values()) + remote.pymolcmdsdict()
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

class PrettyProteinProjectPymolPluginPanel:
    def __init__(self, state):
        self.state = state

    def init_session(self):
        self.polls = Polls(self.state)
        self.toggles = ToggleCommands(self.state)
        pymol.cmd.save(SESSION_RESTORE)
        self.setup_main_window()
        self.update_opts()
        self.toggles.init_session(self.widget.toggles)
        self.polls.init_session(self.widget.polls)
        self.toggles.widget.installEventFilter(self.widget)
        self.polls.widget.installEventFilter(self.widget)

    def setup_main_window(self):
        class ContextDialog(pymol.Qt.QtWidgets.QDialog):
            def eventFilter(self2, source, event):
                if event.type() == pymol.Qt.QtCore.QEvent.ContextMenu:
                    if source is self.polls.widget: return self.polls.context_menu(event)
                    if source is self.toggles.widget: return self.toggles.context_menu(event)
                return super().eventFilter(source, event)

        self.widget = ContextDialog()
        uifile = os.path.join(os.path.dirname(__file__), 'gui_grid_main.ui')
        self.widget = pymol.Qt.utils.loadUi(uifile, self.widget)
        self.widget.show()
        for grade in 'superlike like dislike hate'.split():
            getattr(self.widget, grade).clicked.connect(partial(self.grade_pressed, grade))
        self.widget.button_newpoll.clicked.connect(lambda: self.polls.create_poll_start())
        self.widget.button_use_curdir.clicked.connect(lambda: self.polls.create_poll_from_curdir())
        self.widget.button_newopt.clicked.connect(lambda: self.toggles.create_toggle_start())
        # self.widget.button_save.clicked.connect(lambda: self.save_session())
        # self.widget.button_load.clicked.connect(lambda: self.load_session())
        # self.widget.button_restart.clicked.connect(lambda: self.init_session())
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
        self.polls.pollinprogress.record_review(grade, comment=self.widget.comment.toPlainText())

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

def run_local_server(port=54321):
    ipd.dev.lazyimport('fastapi')
    ipd.dev.lazyimport('sqlmodel')
    ipd.dev.lazyimport('uvicorn')
    ipd.dev.lazyimport('ordered_set')
    args = dict(port=port, log='warning', datadir=os.path.expanduser('~/.config/ppp/localserver'), local=True)
    __server_thread = threading.Thread(target=ppp.server.run, kwargs=args, daemon=True)
    __server_thread.start()
    # dialog should cover server start time
    isfalse_notify(False, f"Can't connt to: {state.serveraddr}\nWill try to run a local server.")
    return ppp.PPPClient(f'127.0.0.1:{port}')

def print_status():
    print(ipd.dev.git_status('plugin code status'))
    print(remote.get('/gitstatus/server code status/end'))

def run(_self=None):
    state_defaults = dict(
        reviewed=set(),
        prefetch=7,
        review_action='cp $file $pppdir/$poll/$grade_$filebase',
        do_review_action=False,
        findcmd='',
        findpoll='',
        shuffle=False,
        use_rsync=False,
        hide_invalid=True,
        showallcmds=False,
        pymol_sc_repr='sticks',
        active_cmds=set(),
        activepoll=None,
        activepollindex=0,
        files=set(),
        serveraddr=os.environ.get('PPPSERVER', 'ppp.ipd:12345'),
        user=getpass.getuser(),
    )
    state_types = dict(
        cmds='global',
        activepoll='global',
        polls='global',
        active_cmds='perpoll',
        reviewed='perpoll',
        pymol_view='perpoll',
        serveraddr='global',
        user='global',
    )
    os.makedirs(os.path.dirname(SESSION_RESTORE), exist_ok=True)
    os.makedirs(os.path.dirname(PPPPP_PICKLE), exist_ok=True)
    if os.path.exists(SESSION_RESTORE): os.remove(SESSION_RESTORE)
    pymol.cmd.do('from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin '
                 'import ppp_pymol_get, ppp_pymol_set, ppp_pymol_add_default')
    global ppppp, remote, state
    state = ipd.dev.StateManager(CONFIG_FILE, STATE_FILE, state_types, state_defaults)
    print(f'user: {state.user}')
    try:
        remote = ppp.PPPClient(state.serveraddr)
    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectionError):
        remote = run_local_server()
    print_status()
    ppppp = PrettyProteinProjectPymolPluginPanel(state)
    ppppp.init_session()

def main():
    print('RUNNING main if for debugging only!')

    while True:
        sleep(0.1)

if __name__ == '__main__':
    main()
