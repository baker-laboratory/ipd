import os
import tempfile
from pathlib import Path

import pytest
from assertpy import assert_that as at

import ipd

def main():
    # run_plugin()
    with tempfile.TemporaryDirectory() as td:
        test_state_manager(td)
    print('test_state DONE', flush=True)

@pytest.mark.fast
def test_state_manager(tmpdir):  # sourcery skip: extract-duplicate-method
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
    conffile = Path(os.path.join(tmpdir, 'conf.yaml'))
    statefile = Path(os.path.join(tmpdir, 'state.yaml'))
    state = ipd.dev.StateManager(conffile, statefile, state_types, state_defaults, debugnames={})
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
    statefile.write_text("""
        active_cmds: !!set {}
        polls:
          poll1:
            foo: bar
          poll2:
            foo: baz
        pymol_view: {}
        """)
    # print(statefile.read_text())
    # print(state._state)
    at(state.foo).is_equal_to('bar')
    state.activepoll = 'poll2'
    at(state.foo).is_equal_to('baz')
    # print(conffile.read_text())
    print(state.polls._special)  # type: ignore
    assert not state.polls.__dict__['_special']['strict_lookup']
    conffile.write_text("""
        cmds: {}
        opts:
            activepoll: poll1
        """)
    at(state.activepoll).is_equal_to('poll1')

    assert isinstance(state.polls, ipd.dev.Bunch)
    assert 'polls' in state._state
    assert 'polls' not in state._conf
    assert state._conf.__dict__['_special']['strict_lookup']
    assert not state._state.__dict__['_special']['strict_lookup']
    assert state._state.polls is state.polls  # type: ignore
    assert not state.polls.__dict__['_special']['strict_lookup']
    assert isinstance(state.polls['foo bar'], ipd.dev.Bunch)

if __name__ == '__main__':
    main()
