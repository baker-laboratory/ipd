import ipd
import os
import yaml
import logging
import time

def add_defaults(server_addr, stress_test_polls=False):
    import pymol
    pymol.cmd.set('suspend_updates', 'on')
    client = ipd.ppp.PPPClient(server_addr)
    if not client.pymolcmds():
        add_builtin_cmds(client)
        add_sym_cmds(client)
    # for pref in ['DUMMY']:
    if stress_test_polls:
        dirs = find_pdb_dirs('~', 1, 20)
        dirs.append('/home/sheffler/project/rfdsym/hilvert/pymol_saves')
        dirs.append('/home/sheffler/project/rfdsym/abbas/pymol_saves')
        for dir_ in dirs:
            client.upload(
                ipd.ppp.PollSpec(name=dir_.replace('/home/sheffler/', '/').replace('/', '__'),
                                 path=dir_,
                                 public=True))
    pymol.cmd.set('suspend_updates', 'off')

def add_builtin_cmds(client):
    with open(__file__.replace('.py', '.yaml')) as inp:
        config = yaml.load(inp, yaml.Loader)
        for cmd in config['pymolcmds']:
            spec = ipd.ppp.PymolCMDSpec(**cmd)
            spec.check_cmds()
            if not spec.errors():
                client.upload(spec)
            else:
                print(spec.errors())
                assert 0, 'cmd errors'

def add_sym_cmds(client):
    for sym in 'c2 c3 c4 c5 c6 d2 d3 tet oct icos'.split():
        # for sym in 'c2 c3 c4 c5 c6 c7 c8 c9 d2 d3 d4 d5 d6 tet oct icos'.split():
        client.upload(
            ipd.ppp.PymolCMDSpec(
                name=f'sym: Make {sym.upper()}',
                cmdstart='from wills_pymol_crap import symgen',
                cmdon=
                f'symgen.make{sym}("$subject", name="sym"); delete $subject; cmd.set_name("sym", "$subject")',
                cmdoff='remove not chain A',
            ))

def find_pdb_dirs(path, lb, ub):
    dirs = []
    for dir_, _, files in os.walk(os.path.expanduser(path)):
        pdbfiles = [f for f in files if f.endswith('.pdb') and os.path.basename(f)[0] != '_']
        if lb <= len(pdbfiles) <= ub:
            dirs.append(dir_)
    return dirs
    # time.sleep(0.01)
