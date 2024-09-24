import ipd
import os
import yaml
import logging
import time

def add_defaults(server_addr):
    import pymol
    pymol.cmd.set('suspend_updates', 'on')
    client = ipd.ppp.PPPClient(server_addr)
    add_builtin_cmds(client)
    add_sym_cmds(client)
    # for pref in ['DUMMY']:
    # for pref in 'abcdefghijklmnop':
        # recursive_add_polls(client, '~', pref)
        # break
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

def recursive_add_polls(client, path, prefix):
        for dir_, _, files in os.walk(os.path.expanduser(path)):
            pdbfiles = [f for f in files if f.endswith('.pdb') and os.path.basename(f)[0] != '_']
            if 10 <= len(pdbfiles) <= 100:
                client.upload(ipd.ppp.PollSpec(name=prefix+dir_[-20:].replace('/', '__'), path=dir_, public=True))
                # time.sleep(0.01)

