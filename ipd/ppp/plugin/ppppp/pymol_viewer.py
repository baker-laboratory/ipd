import os

import pymol

class SubjectName:
    def __init__(self):
        self.count = 0
        self.name = 'subject'

    def __call__(self):
        return f'{self.name}_{self.count}'

    def new(self, name='subject'):
        self.count += 1
        for sfx in ipd.STRUCTURE_FILE_SUFFIX:
            name = name.replace(sfx, '')
        self.name = os.path.basename(name)
        return self()

subject_name = SubjectName()

class PymolPollFileViewer:
    def __init__(self, state, fname, name, toggles):
        self.state = state
        self.fname = fname
        self.toggles = toggles
        pymol.cmd.delete(subject_name())
        pymol.cmd.load(self.fname, subject_name.new(name))
        pymol.cmd.color('green', f'{subject_name()} and elem C')
        self.update()

    def update(self):
        pymol.cmd.reset()
        for cmd in self.state.active_cmds:
            assert cmd in ppppp.toggles.cmds
            self.run_command(ppppp.toggles.cmds[cmd].cmdon)
        if self.state.pymol_view: pymol.cmd.set_view(self.state.pymol_view)

    def update_command(self, toggle: 'ToggleCommand'):
        if toggle: self.run_command(toggle.cmdon)
        else: self.run_command(toggle.cmdoff)

    def run_command(self, cmd: str):
        assert isinstance(cmd, str)
        pymol.cmd.do(cmd.replace('$subject', subject_name()))

    def cleanup(self):
        pymol.cmd.delete(subject_name())
        self.state.pymol_view = pymol.cmd.get_view()
