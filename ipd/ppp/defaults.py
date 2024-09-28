import ipd
import os
import yaml
import logging
import sys
import time
from rich import print

def add_defaults(server_addr, stress_test_polls=False):
    print('---------------------- ADD DEFAULTS ----------------------')
    import pymol
    pymol.cmd.set('suspend_updates', 'on')
    pymol.cmd.do('from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin '
                 'import ppp_pymol_get, ppp_pymol_set, ppp_pymol_add_default')
    client = ipd.ppp.PPPClient(server_addr)
    add_builtin_cmds(client)
    add_sym_cmds(client)
    if stress_test_polls: add_stresstest_polls(client)
    pymol.cmd.set('suspend_updates', 'off')
    print('------------------- DONE ADD DEFAULTS -------------------')

def add_stresstest_polls(client):
    print('add_stresstest_polls start')
    if os.path.exists('/tmp/add_stresstest_polls.list'):
        dirs = [l.strip() for l in open('/tmp/add_stresstest_polls.list').readlines()]
    else:
        dirs = find_pdb_dirs('~', 1, 100)
        dirs.insert(0, '/home/sheffler/project/rfdsym/hilvert/pymol_saves')
        dirs.insert(0, '/home/sheffler/project/rfdsym/abbas/pymol_saves')
        with open('/tmp/add_stresstest_polls.list', 'w') as out:
            out.write(os.linesep.join(dirs))
    presentpolls = {p[1] for p in client.pollinfo()}
    print('add_stresstest_polls', len(dirs), len(presentpolls))
    for dir_ in dirs:
        name = dir_.replace('/home/sheffler/', '').replace('/', ' ').title()
        if name in presentpolls:
            print('skip', name)
            continue
        pollspec = ipd.ppp.PollSpec(name=name, path=dir_, public=True)
        if err := client.upload(pollspec):
            print('create poll failed:', err)
        else:
            print(pollspec)

def add_builtin_cmds(client):
    with open(__file__.replace('.py', '.yaml')) as inp:
        config = yaml.load(inp, yaml.Loader)
        for cmd in config['pymolcmds']:
            spec = ipd.ppp.PymolCMDSpec(**cmd)
            if not spec.errors():
                client.upload(spec, replace=True)
            else:
                print(spec.errors())
                assert 0, 'cmd errors'

def add_sym_cmds(client):
    for sym in 'c2 c3 c4 c5 c6 c7 c8 c9 d2 d3 d4 d5 d6 tet oct icos'.split():
        cmd = ipd.ppp.PymolCMDSpec(
            name=f'sym: Make {sym.upper()}',
            cmdstart='from wills_pymol_crap import symgen',
            cmdon=f'symgen.make{sym}("$subject", name={sym}; delete $subject; cmd.set_name({sym}, "$subject")',
            cmdoff='remove not chain A',
            sym=sym)
        client.upload(cmd, replace=True)

def find_pdb_dirs(path, lb, ub):
    dirs = []
    for dir_, _, files in os.walk(os.path.expanduser(path)):
        if dir_.endswith('traj'): continue
        pdbfiles = [f for f in files if f.endswith('.pdb') and os.path.basename(f)[0] != '_']
        if lb <= len(pdbfiles) <= ub:
            dirs.append(dir_)
    return dirs
    # time.sleep(0.01)
