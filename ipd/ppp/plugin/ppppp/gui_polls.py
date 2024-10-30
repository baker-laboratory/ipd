import duration as timedelta
import os
import random
import subprocess
import traceback
from subprocess import check_output

import pymol

import ipd
from ipd import ppp
from ipd.dev.qt import MenuAction, isfalse_notify, notify

class PollInProgress:
    def __init__(self, root, state, remote, poll):
        self.poll = poll
        self.root = root
        self.state = state
        self.remote = remote
        self.viewer = None
        self.state.activepoll = poll.name
        Cache = ipd.dev.PrefetchLocalFileCache if self.state.prefetch else ipd.dev.FileCache
        ipd.dev.qt.isfalse_notify(self.fnames, f'Poll {self.poll.name} {self.poll.path} has no files')
        self.filecache = Cache(self.fnames, numprefetch=7 if self.state.prefetch else 0)
        self.root.toggles.update_commands_gui()
        self.root.set_pbar(lb=0, val=len(self.state.reviewed), ub=len(self.fnames) - 1)

    def init_files(self):
        fnames = [f.fname for f in self.poll.pollfiles]
        if self.state.shuffle: self.pbdlist = random.shuffle(fnames)
        return fnames

    @property
    def fnames(self):
        if not self.state.fnames: self.state.fnames = self.init_files()
        return self.state.fnames

    @property
    def index(self):
        if not self.state.activepollindex: self.state.activepollindex = 0
        return self.state.activepollindex

    @index.setter
    def index(self, index):
        self.state.activepollindex = index

    def start(self):
        self.root.update_opts()
        self.switch_to(self.index)

    def switch_to(self, index=None, delta=None):
        if index is None: index = self.index
        if delta: index = (index + delta) % len(self.fnames)
        if index >= len(self.fnames): return False
        if self.viewer: self.viewer.cleanup()
        self.root.widget.showfile.setText(self.fnames[index])
        self.viewer = ppp.plugin.PymolPollFileViewer(self.state, self.filecache[index], self.fnames[index],
                                                     self.root.toggles)
        self.index = index
        return True

    def record_review(self, grade, comment):
        review = ppp.ReviewSpec(grade=grade,
                                comment=comment,
                                pollid=self.poll.id,
                                fname=self.fnames[self.index])
        if self.state.do_review_action and not self.exec_review_action(review): return
        response = self.remote.upload_review(review, self.fnames[self.index])
        if isfalse_notify(not response, f'upload file server response: {response}'): return
        self.review_accepted(review)

    def review_accepted(self, review):
        pymol.cmd.delete(subject_name())
        self.state.reviewed.add(self.viewer.fname)
        self.root.set_pbar(lb=0, val=len(self.state.reviewed), ub=len(self.fnames) - 1)
        self.root.widget.comment.setText('')
        if len(self.state.reviewed) == len(self.fnames): self.root.polls.poll_finished()
        else: self.switch_to(delta=1)

    def preprocess_shell_cmd(self, cmd):
        cmd = cmd.replace('$pppdir', os.path.abspath(os.path.expanduser(self.state.pppdir)))
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
                    for fn in filter(lambda s: s.endswith(ipd.STRUCTURE_FILE_SUFFIX), noopt[1:-1]):
                        assert os.path.exists(fn)
                cmds.append(c.split())
        return cmds

    def exec_review_action(self, review):
        if review.grade in ('dislike', 'hate'): return True
        cmds = self.preprocess_shell_cmd(self.state.review_action.replace('$grade', review.grade))
        for cmd in cmds:
            try:
                check_output(cmd, stderr=subprocess.STDOUT)
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
        self.root.widget.showfile.setText('')
        self.root.widget.showsym.setText('')
        self.root.widget.showlig.setText('')
        self.filecache.cleanup()

class Polls(ipd.dev.qt.ContextMenuMixin):
    def __init__(self, root, state, remote):
        self.root = root
        self.state = state
        self.remote = remote
        self.pollinprogress = None
        self.current_poll_index = None
        self.listitems = None
        self.Qt = pymol.Qt

    def init_session(self, widget):
        self.widget = widget
        self._install_event_filter(self.root.widget)
        self.widget.itemClicked.connect(lambda a: self.poll_clicked(a))
        self.newpollwidget = pymol.Qt.QtWidgets.QDialog()
        uifile = os.path.join(os.path.dirname(__file__), 'gui_new_poll.ui')
        self.newpollwidget = pymol.Qt.utils.loadUi(uifile, self.newpollwidget)
        self.newpollwidget.openfiledialog.clicked.connect(lambda: self.open_file_picker())
        self.newpollwidget.cancel.clicked.connect(lambda: self.newpollwidget.hide())
        self.newpollwidget.autofill.clicked.connect(lambda: self.create_poll_autofill_button())
        self.refresh_polls()

    def open_file_picker(self, public=None):
        dialog = pymol.Qt.QtWidgets.QPollFileDialog(self.newpollwidget)
        dialog.setPollFileMode(pymol.Qt.QtWidgets.QPollFileDialog.Directory)
        dialog.setDirectory(os.path.expanduser('~'))
        dialog.show()
        if dialog.exec_():
            file_names = dialog.selectedPollFiles()
            assert len(file_names) == 1
            if public is not None: self.newpollwidget.ispublic.setCheckState(public * 2)
            self.newpollwidget.path.setText(file_names[0])
            self.newpollwidget.show()

    def _context_menu_items(self):
        return {
            'details': MenuAction(func=lambda cmd: ipd.dev.qt.notify(cmd)),
            'refersh': MenuAction(func=self.refresh_polls, item=False),
            'edit': MenuAction(func=self.edit_poll_start, owner=True),
            'clone': MenuAction(func=self.clone_poll_start),
            'delete': MenuAction(func=self.delete_poll, owner=True),
            'shuffle on/off': MenuAction(func=self.shuffle_poll, item=True),
        }

    def _get_from_item(self, item):
        return self.remote.poll(id=self.allpolls[item.text()])

    def shuffle_poll(self, poll):
        if poll: self.state.polls[poll.name].shuffle = not self.state.polls[poll.name].shuffle
        else: self.state._conf.shuffle = not self.state._conf.shuffle

    def edit_poll_start(self, poll):
        self.update_gui_from_poll(poll)
        self.newpollwidget.title.setText('Edit Poll')
        self.newpollwidget.ok.setText('Edit Poll')
        self.newpollwidget.ok.clicked.connect(lambda: self.edit_poll_done(poll))
        self.newpollwidget.show()

    def edit_poll_done(self, origpoll):
        poll = self.create_poll_spec_from_gui()
        if self.create_poll(poll):
            self.newpollwidget.hide()
            self.delete_poll(origpoll)

    def clone_poll_start(self, poll):
        self.update_gui_from_poll(poll)
        self.newpollwidget.name.setText(f'Clone of {poll.name}')
        self.newpollwidget.title.setText('Clone Poll')
        self.newpollwidget.show()
        self.newpollwidget.ok.clicked.connect(lambda: self.create_poll_ok_button())

    def delete_poll(self, poll):
        if self.state.activepoll == poll.name: self.state.activepoll = None
        self.remote.remove(poll)
        if len(self.listitems) == 1: self.listitems.pop().setSelected(False)
        self.refresh_polls()

    def refresh_polls(self):
        # localpolls = [(p.id, p.name, p.user, p.desc, p.sym, p.ligand) for p in self.state.local.polls.values()]
        self.pollsearchtext, self.polltooltip, allpolls = [], {}, {}
        self.listitems, self.listitemdict = [], {}
        self.allpolls = self.remote.pollinfo(user=self.state.user)  #+ localpolls
        if not self.allpolls: return
        print(self.remote.polls(_ghost=True))
        print('pollinfo', self.allpolls)
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
        if self.state.activepoll and self.state.activepoll in self.allpolls:
            self.poll_start(self.state.activepoll)

    def filtered_poll_list(self):
        polls = set(self.allpolls)
        if query := self.state.findpoll:
            from subprocess import PIPE, Popen
            p = Popen(['fzf', '-i', '--filter', f'{query}'], stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
            hits = p.communicate(input=self.pollsearchtext)[0]
            hits = [m[:m.find('||||')] for m in hits.split('\n') if m]
            polls = set(hits)
            if self.pollinprogress: polls.add(self.pollinprogress.poll.name)
        return polls

    def update_polls_gui(self):
        if self.listitems is None: self.refresh_polls()
        self.visiblepolls = self.filtered_poll_list()
        if self.state.activepoll in self.listitemdict:
            self.listitemdict[self.state.activepoll].setSelected(True)
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
        poll = self.remote.poll(id=self.allpolls[name])
        self.pollinprogress = PollInProgress(self.root, self.state, self.remote, poll)
        self.state.activepoll = self.pollinprogress.poll.name
        self.root.toggles.update_commands_gui()
        self.pollinprogress.start()

    def poll_finished(self):
        if self.pollinprogress: self.pollinprogress.cleanup()
        self.pollinprogress = None
        self.state.activepoll = None
        self.root.update_opts()
        self.root.set_pbar(done=True)
        # self.update_polls_gui()

    def create_poll_start(self):
        self.newpollwidget.user.setText(self.state.user)
        self.newpollwidget.show()
        self.newpollwidget.ok.clicked.connect(lambda: self.create_poll_ok_button())
        self.newpollwidget.ok.setText('Ok')
        self.newpollwidget.title.setText('Create New Poll')

    def create_poll_spec_from_gui(self):
        # print('create_poll_spec_from_gui')
        # sourcery skip: dict-assign-update-to-union
        duration = ipd.dev.safe_eval('dict(' + ','.join(self.newpollwidget.duration.text().split()) + ')')
        duration = timedelta(**duration)
        duration = duration or timedelta(weeks=99999)
        # if isfalse_notify(self.newpollwidget.name.text(), 'Must provide a Name'): return
        # if isfalse_notify(os.path.exists(os.path.expanduser(self.newpollwidget.path.text())),
        # 'path must exist'):
        # return
        if isfalse_notify(duration > timedelta(minutes=1), 'Poll expires too soon'): return
        fields = 'name path sym ligand user workflow cmdstart cmdstop props attrs'
        kw = {k: ipd.dev.qt.widget_gettext(getattr(self.newpollwidget, k)) for k in fields.split()}
        kw |= {k: bool(getattr(self.newpollwidget, k).checkState()) for k in 'ispublic telemetry'.split()}
        kw['enddate'] = datetime.now() + duration
        return self.create_poll_spec(**kw)

    def update_gui_from_poll(self, poll):
        for k in 'name path sym ligand user cmdstart cmdstop props attrs nchain'.split():
            val = str(poll[k]) if poll[k] else ''
            ipd.dev.qt.widget_settext(getattr(self.newpollwidget, k), val)
        for k in 'ispublic telemetry'.split():
            getattr(self.newpollwidget, k).setCheckState(2 * poll[k])

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
        response = self.remote.upload_poll(poll)
        if isfalse_notify(not response, f'server response: {response}'): return False
        self.refresh_polls()
        return True

    def create_poll_spec(self, **kw):
        kw['userid'] = kw['user']
        del kw['user']
        try:
            spec = ppp.PollSpec(**kw)
            assert isinstance(spec, ppp.PollSpec)
            return spec
        except Exception as e:
            notify(f'create PollSpec error:\n{e}\n{traceback.format_exc()}')
            return None

    def create_poll_from_curdir(self):
        u = self.state.user
        d = os.path.abspath('.').replace(f'/mnt/home/{u}', '~').replace(f'/home/{u}', '~')
        self.create_poll(
            self.create_poll_spec(
                name=f"PollFiles in {d.replace('/',' ')} ({u})",
                desc='The poll for lazy people',
                path=d,
                user=self.state.user,
                ispublic=False,
                telemetry=False,
            ))

    def cleanup(self):
        if self.pollinprogress: self.pollinprogress.cleanup()
