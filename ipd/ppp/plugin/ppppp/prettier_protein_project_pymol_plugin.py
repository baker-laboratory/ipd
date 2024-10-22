import collections
import os
import sys
import threading
import time
from functools import partial

import pymol
from rich import print

import ipd
from ipd import ppp
from ipd.dev.qt import isfalse_notify

it = ipd.lazyimport('itertools', 'more_itertools', pip=True)
requests = ipd.lazyimport('requests', pip=True)
# fuzzyfinder = ipd.lazyimport('fuzzyfinder', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
wpc = ipd.lazyimport('wills_pymol_crap', 'git+https://github.com/willsheffler/wills_pymol_crap.git', pip=True)
ipd.h
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
    state._defaults[name] = val
    if state: state._statetype[name] = 'global' if isglobal else 'perpoll'

def ppp_pymol_get(name):
    if not ppppp: return TEST_STATE[name]
    # print('ppp get', state[f'ppp_pymol_{name}'])
    return state[f'ppp_pymol_{name}']

def ppp_pymol_set(name, val):
    # print('PYMOLSET', name, val, ppppp)
    if not ppppp: TEST_STATE[name] = val
    else: state[f'ppp_pymol_{name}'] = val

class PrettyProteinProjectPymolPluginPanel:
    def __init__(self, state, remote):
        self.state = state
        self.remote = remote

    def init_session(self):
        pymol.cmd.save(SESSION_RESTORE)
        self.setup_main_window()
        self.toggles = ppp.plugin.ToggleCommands(self, self.state, self.remote)
        self.polls = ppp.plugin.Polls(self, self.state, self.remote)
        self.update_opts()
        self.toggles.init_session(self.widget.toggles)
        self.polls.init_session(self.widget.polls)
        self.flowcreator = ppp.plugin.WorkflowCreatorGui(self, self.state, self.remote)
        self.setup_keybinds()

    def setup_main_window(self):
        class ContextDialog(pymol.Qt.QtWidgets.QDialog):
            def eventFilter(self2, source, event):
                if event.type() == pymol.Qt.QtCore.QEvent.ContextMenu:
                    if source is self.polls.widget: return self.polls.context_menu(event)
                    if source is self.toggles.widget: return self.toggles.context_menu(event)
                return super().eventFilter(source, event)

        uifile = os.path.join(os.path.dirname(__file__), 'gui_grid_main.ui')
        self.widget = pymol.Qt.utils.loadUi(uifile, ContextDialog())
        self.widget.show()
        for grade in 'superlike like dislike hate'.split():
            getattr(self.widget, grade).clicked.connect(partial(self.grade_pressed, grade))
        self.widget.button_newpoll.clicked.connect(lambda: self.polls.create_poll_start())
        self.widget.button_use_curdir.clicked.connect(lambda: self.polls.create_poll_from_curdir())
        self.widget.button_use_dir.clicked.connect(lambda: self.polls.open_file_picker(public=0))
        self.widget.button_newopt.clicked.connect(lambda: self.toggles.create_command_start())
        self.widget.button_newflow.clicked.connect(lambda: self.flowcreator.create_flow_start())
        # self.widget.button_save.clicked.connect(lambda: self.save_session())
        # self.widget.button_load.clicked.connect(lambda: self.load_session())
        # self.widget.button_restart.clicked.connect(lambda: self.init_session())
        self.widget.button_quit.clicked.connect(lambda: self.quit())
        self.widget.button_quitpymol.clicked.connect(lambda: self.quit(exitpymol=True))

    def setup_keybinds(self):
        pymol.cmd.set_key('pgup', lambda: self.polls.pollinprogress.switch_to(delta=-1))
        pymol.cmd.set_key('pgdn', lambda: self.polls.pollinprogress.switch_to(delta=1))
        pymol.cmd.set_key('F1', partial(self.grade_pressed, 'superlike'))
        pymol.cmd.set_key('F2', partial(self.grade_pressed, 'like'))
        pymol.cmd.set_key('F3', partial(self.grade_pressed, 'dislike'))
        pymol.cmd.set_key('F4', partial(self.grade_pressed, 'hate'))

    def update_opts(self):
        # print('UPDATE OPTS', ppppp.polls.pollinprogress)
        action = collections.defaultdict(lambda: lambda: None)
        action['hide_invalid'] = self.polls.update_polls_gui
        action['showallcmds'] = self.toggles.update_commands_gui
        action['findpoll'] = self.polls.update_polls_gui
        action['findcmd'] = self.toggles.update_commands_gui
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

def run(_self=None):
    from ipd.ppp.plugin.ppppp.ppppp_defaults import state_defaults, state_types
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
    # print(ipd.dev.git_status('plugin code status'))
    print(remote.get('/gitstatus/server code status/end'))
    ppppp = PrettyProteinProjectPymolPluginPanel(state, remote)
    ppppp.init_session()

def main():
    print('RUNNING main if for debugging only!')

    while True:
        time.sleep(0.1)

if __name__ == '__main__':
    main()
