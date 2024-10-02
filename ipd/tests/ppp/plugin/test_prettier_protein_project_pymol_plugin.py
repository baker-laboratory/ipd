import tempfile
import os
import json
import time
from pathlib import Path
from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin import *
import ipd

at = ipd.lazyimport('assertpy', pip=True).assert_that

def main():
    # run_plugin()
    with tempfile.TemporaryDirectory() as td:
        test_state_manager(td)
    run_pymol()
    print('test_ppppp DONE', flush=True)

def test_state_manager(tempdir):  # sourcery skip: extract-duplicate-method
    conffile = Path(os.path.join(tempdir, 'conf.yaml'))
    statefile = Path(os.path.join(tempdir, 'state.yaml'))
    state = StateManager(conffile, statefile, debugnames={})
    assert 'foo' not in state
    state.foo = 'fooglobalstate'
    at(state.foo).is_equal_to('fooglobalstate')
    state.activepoll = 'poll1'
    at(state.foo).is_equal_to(None)
    state.foo = 'foopoll1'
    at(state.foo).is_equal_to('foopoll1')
    state.activepoll = 'poll2'
    at(state.foo).is_equal_to(None)
    state.foo = 'foopoll2'
    at(state.foo).is_equal_to('foopoll2')
    state.activepoll = 'poll1'
    at(state.foo).is_equal_to('foopoll1')

    s = statefile.read_text()
    assert s.count('poll1') and s.count('poll2')
    assert not s.count('activepoll')
    statefile.write_text('''
        active_cmds: !!set {}
        polls:
          poll1:
            foo: bar
          poll2:
            foo: baz
        pymol_view: {}
        ''')
    # print(statefile.read_text())
    # print(state._state)
    at(state.foo).is_equal_to('bar')
    state.activepoll = 'poll2'
    at(state.foo).is_equal_to('baz')
    # print(conffile.read_text())
    print(state.polls._special)
    assert not state.polls.__dict__['_special']['strict_lookup']
    conffile.write_text('''
        cmds: {}
        opts:
            activepoll: poll1
        ''')
    at(state.activepoll).is_equal_to('poll1')

    assert isinstance(state.polls, ipd.Bunch)
    assert 'polls' in state._state
    assert 'polls' not in state._conf
    assert state._conf.__dict__['_special']['strict_lookup']
    assert not state._state.__dict__['_special']['strict_lookup']
    assert state._state.polls is state.polls
    assert not state.polls.__dict__['_special']['strict_lookup']
    assert isinstance(state.polls['foo bar'], ipd.Bunch)

def run_polls_stress_test():
    server, backend = ipd.ppp.server.run(12345, 'postgresql://sheffler@192.168.0.154:5432/ppp')
    client = ipd.ppp.PPPClient('127.0.0.1:12345')
    polls = client.pollinfo()
    print(len(polls))
    print(polls[0])
    ipd.dev.global_timer.report()
    server.stop()

def run_plugin():
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    run()
    print('^' * 100)
    setstate('prefetch', 0)
    at(0).is_equal_to(getstate('prefetch'))
    setstate('prefetch', 2)
    at(2).is_equal_to(getstate('prefetch'))

def run_pymol():
    # os.environ['QT_QPA_PLATFORM'] = 'xcb'
    pymol.pymol_argv = ['pymol', '-q']
    pymol.finish_launching()
    # ipd.ppp.plugin.ppppp.run()
    # from ipd.ppp.plugin.ppppp import run_ppppp_gui
    # ui = run_ppppp_gui()
    # while time.sleep(1): pass
    # assert 0
    # from ipd.pymol import ppppp

if __name__ == '__main__':
    main()
