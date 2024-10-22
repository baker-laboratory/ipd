import os

import pymol

import ipd
from ipd.dev.qt import MenuAction, isfalse_notify

_startup_cmds_done = set()

def ensure_startup_cmd(cmd):
    if cmd in _startup_cmds_done: return
    pymol.cmd.do(cmd)
    if any([
            cmd.startswith(('import ', 'from ', 'ppp_pymol_add_default')),
    ]):
        _startup_cmds_done.add(cmd)

class ToggleCommand(ipd.ppp.PymolCMD):
    def __init__(self, widget, root, **kw):
        super().__init__(root.remote, **kw)
        self._root, self._widget = root, widget
        if self.cmdstart: ensure_startup_cmd(self.cmdstart)
        assert self.widget.text().startswith(self.name)
        # self.widget.setCheckState(2 * int(self.onstart))

    @property
    def widget(self):
        return self._widget

    def widget_update(self, toggle=False):
        if toggle:
            self.widget.setCheckState(0 if self.widget.checkState() else 2)
            if self.widget.checkState(): self._root.state.active_cmds.add(self.name)
            else: self._root.state.active_cmds.remove(self.name)
        if self._root.polls.pollinprogress and self._root.polls.pollinprogress.viewer:
            # print('toggle update')
            self._root.polls.pollinprogress.viewer.update_command(toggle=self)

    def __bool__(self):
        return bool(self.widget.checkState())

class ToggleCommands(ipd.dev.qt.ContextMenuMixin):
    def __init__(self, parent, state, remote):
        self.name = self.__class__.__name__
        self.parent = parent
        self.state = state
        self.remote = remote
        self.Qt = pymol.Qt
        self.widget = None
        self.itemsdict = None
        self.cmds = {}

    def init_session(self, widget):
        self.widget = widget
        self._install_event_filter(self.parent.widget)
        self.refresh_command_list()
        # widget.itemChanged.connect(lambda _: self.update_item(_))
        self.widget.itemClicked.connect(lambda _: self.update_item(_, toggle=True))
        self.newcmdwidget = pymol.Qt.QtWidgets.QDialog()
        self.newcmdwidget = pymol.Qt.utils.loadUi(
            os.path.join(os.path.dirname(__file__), 'gui_new_pymolcmd.ui'), self.newcmdwidget)
        self.newcmdwidget.cancel.clicked.connect(lambda: self.newcmdwidget.hide())

    def _context_menu_items(self):
        return {
            'details': MenuAction(func=lambda cmd: ipd.dev.qt.notify(cmd)),
            'refersh': MenuAction(func=self.refresh_command_list, item=False),
            'edit': MenuAction(func=self.edit_command_start, owner=True),
            'clone': MenuAction(func=self.clone_command_start),
            'delete': MenuAction(func=self.delete_command, owner=True),
        }

    def _get_from_item(self, item):
        return self.cmds[item.text()]

    def edit_command_start(self, toggle):
        self.update_gui_from_cmd(toggle)
        self.newcmdwidget.title.setText('Edit Command')
        self.newcmdwidget.ok.setText('Edit Command')
        self.newcmdwidget.ok.clicked.connect(lambda: self.edit_command_done(toggle))
        self.newcmdwidget.show()

    def edit_command_done(self, origcmd):
        toggle = self.create_poll_spec_from_gui()
        if self.create_poll(toggle):
            self.delete_command(origcmd)

    def clone_command_start(self, cmd):
        self.update_gui_from_cmd(cmd)
        self.newcmdwidget.name.setText(f'Clone of {cmd.name}')
        self.newcmdwidget.title.setText('Clone Command')
        self.newcmdwidget.show()
        self.newcmdwidget.ok.clicked.connect(lambda: self.create_command_done())

    def delete_command(self, toggle):
        self.remote.remove(toggle)
        self.refresh_command_list()

    def update_item(self, item, toggle=False):
        self.cmds[item.text()].widget_update(toggle)

    def create_command_start(self):
        self.newcmdwidget.user.setText(self.state.user)
        self.newcmdwidget.title.setText('Create New PymolCMD')
        self.newcmdwidget.ok.setText('Create CMD')
        self.newcmdwidget.show()
        self.newcmdwidget.ok.clicked.connect(lambda: self.create_command_done())

    def update_gui_from_cmd(self, cmd):
        for k in 'name cmdon cmdoff cmdstart sym ligand props attrs'.split():
            val = str(cmd[k]) if cmd[k] else ''
            ipd.dev.qt.widget_settext(getattr(self.newcmdwidget, k), val)
        for k in 'ispublic onstart'.split():
            getattr(self.newcmdwidget, k).setCheckState(2 * cmd[k])

    def create_cmdspec_from_gui(self):  # sourcery skip: dict-assign-update-to-union
        if isfalse_notify(self.newcmdwidget.name.text(), 'Must provide a Name'): return
        if isfalse_notify(self.newcmdwidget.cmdon.toPlainText(), 'Must provide a command'): return
        fields = 'name cmdon cmdoff cmdstart sym ligand props attrs'
        kw = {k: ipd.dev.qt.widget_gettext(getattr(self.newcmdwidget, k)) for k in fields.split()}
        kw |= {k: bool(getattr(self.newcmdwidget, k).checkState()) for k in 'ispublic onstart'.split()}
        return ipd.ppp.PymolCMDSpec(**kw)

    def create_command_done(self):
        cmdspec = self.create_cmdspec_from_gui()
        if isfalse_notify(not cmdspec.errors(), cmdspec.errors()): return
        if cmdspec.ispublic:
            result = self.remote.upload(cmdspec)
            assert not result, result
        else:
            cmd = ipd.ppp.PymolCMD(None, id=len(self.state.cmds) + 1, **cmdspec.model_dump())
            setattr(self.state.cmds, cmd.name, cmd)
        self.newcmdwidget.hide()
        self.refresh_command_list()
        return True

    def refresh_command_list(self):
        assert self.widget is not None
        cmds = self.remote.pymolcmds()
        # print([c['name'] for c in cmdsdicts])
        if 'active_cmds' not in self.state:
            self.state.active_cmds = {cmd.name for cmd in cmds if cmd.onstart}
        self.itemsdict = {}
        self.widget.clear()
        for cmd in cmds:
            self.widget.addItem(cmd['name'])
            item = self.widget.item(self.widget.count() - 1)
            item.setFlags(item.flags() | pymol.Qt.QtCore.Qt.ItemIsUserCheckable)
            self.itemsdict[cmd.name] = item
            cmd = ToggleCommand(item, self.parent, **cmd.model_dump())
            item.setToolTip(
                f'NAME: {cmd.name}\nON: {cmd.cmdon}\nOFF: {cmd.cmdoff}\nNCHAIN: {cmd.minchains}-{cmd.maxchains}'
                f'\nispublic: {cmd.ispublic}\nSYM: {cmd.sym}\nLIG:{cmd.ligand}\nDBKEY:{cmd.id}')
            cmd.widget.setCheckState(2) if cmd.name in self.state.active_cmds else cmd.widget.setCheckState(0)
            self.cmds[cmd.name] = cmd
        self.cmdsearchtext = '\n'.join(f'{c.name}||||{c.desc} sym:{c.sym} user:{c.user.name} lig:{c.ligand}'
                                       for c in self.cmds.values())
        self.update_commands_gui()

    def filtered_cmd_list(self):
        hits = set(self.cmds.keys())
        if query := self.state.findcmd:
            from subprocess import PIPE, Popen
            p = Popen(['fzf', '-i', '--filter', f'{query}'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
            hits = p.communicate(input=self.cmdsearchtext)[0]
            hits = [m[:m.find('||||')] for m in hits.split('\n') if m]
        if not self.state.showallcmds and len(hits) < 512:
            cmds, sym, ligand, nchain = self.cmds, '', '', -1
            if pollip := self.parent.polls.pollinprogress:
                sym, ligand, nchain = pollip.poll.sym, pollip.poll.ligand, pollip.poll.nchain
            hits = filter(lambda x: cmds[x].sym in ('', sym), hits)
            hits = filter(lambda x: cmds[x].ligand in ligand or (ligand and cmds[x].ligand == 'ANY'), hits)
            if nchain > 0: hits = filter(lambda x: cmds[x].minchains <= nchain <= cmds[x].maxchains, hits)
        return set(hits) | set(self.state.active_cmds)

    def update_commands_gui(self):
        if self.itemsdict is None: self.refresh_command_list()
        visible = {k: self.cmds[k] for k in self.filtered_cmd_list()}
        for name, item in self.itemsdict.items():
            item.setCheckState(2 if name in self.state.active_cmds else 0)
            item.setHidden(not (name in visible or item.isSelected()))

    def cleanup(self):
        pass
