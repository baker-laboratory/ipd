import os
from typing import Any

import pymol

import ipd
from ipd.dev.qt import MenuAction, isfalse_notify
from ipd.ppp.plugin.ppppp.gui_commands import ToggleCommands

class FlowStepGui(ipd.ppp.FlowStepSpec):
    widget: Any

class WorkflowGui(ipd.ppp.WorkflowSpec):
    guisteps: list[FlowStepGui]

class WorkflowStepCmdList(ipd.dev.qt.ContextMenuMixin):
    def __init__(self, creator, step):
        class WFCmdList(pymol.Qt.QtWidgets.QListWidget):
            def __init__(self, cmdlist, creator, step):
                super().__init__()
                self.cmdlist, self.creator, self.step = cmdlist, creator, step
                self.setDragEnabled(True)
                self.setAcceptDrops(True)
                self.setDropIndicatorShown(True)

            def mousePressEvent(self, event):
                if event.type() == pymol.Qt.QtCore.QEvent.MouseButtonPress:
                    if event.button() == pymol.Qt.QtCore.Qt.LeftButton:
                        self.creator.select_step(self.step)
                super().mousePressEvent(event)

            def eventFilter(self, source, event):
                if event.type() == pymol.Qt.QtCore.QEvent.ContextMenu:
                    return self.cmdlist.context_menu(event)
                return super().eventFilter(source, event)

        self.Qt = pymol.Qt
        self.creator, self.step = creator, step
        self.widget = WFCmdList(self, creator, step)
        self._install_event_filter(self.widget)
        self.widget.itemClicked.connect(lambda item: self.step.item_clicked(item))
        self.update_size()

    def update_size(self):
        self.widget.setMinimumSize(100, 15 * self.widget.count())

    def remove(self, cmd):
        row = self.widget.row(cmd._guilistitem)
        assert row >= 0
        self.widget.takeItem(row)
        self.widget.update()

    def _context_menu_items(self):
        return {
            'details': MenuAction(func=lambda cmd: ipd.dev.qt.notify(cmd)),
            'remove cmd': MenuAction(func=lambda item: self.creator.remove_cmd(self.step, item)),
            'remove step': MenuAction(func=lambda: self.creator.remove_step(self.step), item=False),
        }

    def _get_from_item(self, item):
        cmd = self.creator.cmds[item.text()]
        cmd._guilistitem = item
        return cmd

class WorkflowStepGui:
    def __init__(self, creator):
        self.creator = creator
        # uifile = os.path.join(os.path.dirname(__file__), 'gui_workflow_step.ui')
        # stepwidget = pymol.Qt.utils.loadUi(uifile, self.newflowwidget)
        self.widget = pymol.Qt.QtWidgets.QWidget(objectName='box')
        self.widget.setLayout(pymol.Qt.QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(4, 4, 4, 4)
        self.cmds = []
        self.cmdlist = WorkflowStepCmdList(self.creator, self)
        self.widget.layout().addLayout(top := pymol.Qt.QtWidgets.QHBoxLayout())
        top.addWidget(name := pymol.Qt.QtWidgets.QLineEdit('', placeholderText='Step Name'))
        top.addWidget(rem := pymol.Qt.QtWidgets.QPushButton('Remove Step'))
        rem.clicked.connect(lambda: self.creator.remove_step(self))
        self.widget.layout().addWidget(taskline := self.create_taskline_stack())
        self.widget.layout().addWidget(self.cmdlist.widget)
        [self.widget.layout().addStretch(i) for i in [1, 1, 100]]
        name.setContentsMargins(0, 0, 0, 0)

        self.taskline = taskline
        self.taskmaker = None
        self.taskmaker_change(0)
        self.name = name
        name.textChanged.connect(lambda _: self.creator.select_step(self))

    def create_taskline_stack(self):
        self.stack = pymol.Qt.QtWidgets.QStackedWidget()
        self.step_types = {
            'Whole Structure': {},
            'For Each Interface': {},
            'For Interfaces With Chain':
            dict(chain=pymol.Qt.QtWidgets.QLineEdit('A')),
            'For Each Ligand': {},
            'For Specific Ligand':
            dict(lig=pymol.Qt.QtWidgets.QLineEdit('', placeholderText='LIG'),
                 _txt=pymol.Qt.QtWidgets.QLabel('in chain'),
                 chain=pymol.Qt.QtWidgets.QLineEdit('ANY')),
        }
        for name, extras in self.step_types.items():
            dropdown = pymol.Qt.QtWidgets.QComboBox()
            for name in self.step_types:
                dropdown.addItem(name)
            dropdown.setCurrentIndex(0)
            dropdown.currentIndexChanged.connect(lambda i: self.taskmaker_change(i))
            panel = pymol.Qt.QtWidgets.QWidget()
            panel.setLayout(pymol.Qt.QtWidgets.QHBoxLayout())
            panel.layout().setContentsMargins(0, 0, 0, 0)
            # panel.layout().addWidget(pymol.Qt.QtWidgets.QLabel(name))
            panel.layout().addWidget(dropdown)
            panel.dropdown = dropdown
            panel.extras = {k: v for k, v in extras.items() if k[0] != '_'}
            for label, widget in extras.items():
                panel.layout().addWidget(widget)
            self.stack.addWidget(panel)
        return self.stack

    def taskmaker_change(self, index):
        self.stack.setCurrentIndex(index)
        self.stack.currentWidget().dropdown.setCurrentIndex(index)
        self.extras = self.stack.currentWidget().extras
        self.creator.select_step(self)

    def select(self):
        self.widget.setStyleSheet("QWidget#box { border: 2px solid gray; }")
        self.creator.state.active_cmds = self.cmds
        self.creator.selstep = self

    def deselect(self):
        self.widget.setStyleSheet('')

    def item_clicked(self, item):
        if item.checkState() and item.text() not in self.cmds: self.cmds.append(item.text())
        elif not item.checkState() and item.text() in self.cmds: self.cmds.remove(item.text())
        self.creator.update_commands_gui()

class WorkflowCreatorGui(ToggleCommands):
    def __init__(self, parent, state, remote):
        self.sharedstate = state
        state = ipd.dev.Bunch(_strict=False, cmds={}, active_cmds=set(), showallcmds=True, findcmd='')
        super().__init__(parent, state, remote)
        self.name = self.__class__.__name__
        uifile = os.path.join(os.path.dirname(__file__), 'gui_workflows.ui')
        self.newflowwidget = pymol.Qt.QtWidgets.QDialog()
        self.newflowwidget = pymol.Qt.utils.loadUi(uifile, self.newflowwidget)
        self.newflowwidget.button_create.clicked.connect(lambda: self.create_flow_done())
        self.newflowwidget.button_newstep.clicked.connect(lambda: self.new_flow_step())
        self.steps = []
        self.reload_steps()
        self.widget = self.newflowwidget.cmdlist
        self.widget.setDragEnabled(True)
        self.selstep = None
        self.init_session(self.widget)
        self.newflowwidget.findcmd.textChanged.connect(
            lambda _: (setattr(self.state, 'findcmd', _), self.update_commands_gui()))
        self.new_flow_step()

    def new_flow_step(self):
        self.steps.append(new := WorkflowStepGui(self))
        self.guisteps.addWidget(new.widget)
        self.select_step(new)

    def select_step(self, step):
        if self.selstep is step: return
        [step.deselect() for step in self.steps]
        step.select()
        self.update_commands_gui()

    def update_item(self, item, toggle):
        if not self.selstep: self.new_flow_step()
        if item.text() in self.selstep.cmds: return
        newitem = pymol.Qt.QtWidgets.QListWidgetItem(item.text())
        # print('setting is checkable flag')
        newitem.setFlags(newitem.flags() | pymol.Qt.QtCore.Qt.ItemIsUserCheckable)
        newitem.setCheckState(2)
        newitem.setToolTip(item.toolTip())
        self.selstep.cmdlist.widget.addItem(newitem)
        self.selstep.cmdlist.update_size()
        self.selstep.cmds.append(item.text())
        self.update_commands_gui()

    def reload_steps(self):
        stepsw = pymol.Qt.QtWidgets.QWidget()
        guisteps = pymol.Qt.QtWidgets.QVBoxLayout()
        guisteps.setContentsMargins(0, 0, 0, 0)
        stepsw.setLayout(guisteps)
        for step in self.steps:
            guisteps.addWidget(step.widget)
        self.guisteps = guisteps
        self.newflowwidget.scroll.setWidget(stepsw)

    def remove_step(self, step):
        self.steps.remove(step)
        self.reload_steps()
        self.update_commands_gui()

    def remove_cmd(self, step, cmd):
        step.cmdlist.remove(cmd)
        if step == self.selstep: self.state.active_cmds.remove(cmd.name)
        self.update_commands_gui()

    def create_flow_start(self):
        self.newflowwidget.show()
        self.refresh_command_list()

    def flow_from_gui():
        print('flow flow_from_gui')

    def gui_error_check(self):
        if isfalse_notify(self.newflowwidget.name.text(), 'You must provide a name for this workflow'): return
        msg = 'You must provide a description for this workflow'
        if isfalse_notify(self.newflowwidget.desc.toPlainText(), msg): return
        if isfalse_notify(self.guisteps.count(), 'You must provide at least one workflow step'): return
        missing = [i + 1 for i in range(self.guisteps.count()) if not self.steps[i].name.text()]
        if isfalse_notify(not missing, f'Workflow step(s) {missing} have no Name'): return
        for step in self.steps:
            n = step.name.text()
            isfalse_notify(step.cmdlist.widget.count(), f'Step {n} has no cmds')
            for p, w in step.extras.items():
                isfalse_notify(w.text(), f'Step {n} param {p} has no value')
                msg = f'Step {n} invalid chain, should be 1 letter or ANY'
                if p == 'chain': isfalse_notify(len(w.text()) == 1 or w.text().lower() == 'any', msg)
                msg = f'Step {n} invalid ligand code, should be 3 letters'
                if p == 'lig': isfalse_notify(len(w.text()) == 3, msg)
        return True

    def create_flow_done(self):
        print('create flow done')
        if self.gui_error_check(): return
        flow = self.flow_from_gui()
        # isfalse_notify(flow, flow.errors())
        self.newflowwidget.hide()
