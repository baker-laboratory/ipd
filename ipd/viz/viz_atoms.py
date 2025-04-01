import random
import tempfile
import numpy as np
import ipd
from ipd.viz.pymol_viz import lazy_register

def show_atoms_pymol(atoms, name='atoms'):
    tag = str(random.random())[2:]
    atoms = atoms[~np.any(np.isnan(atoms.coord), axis=1)]
    pymol_chains = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz")
    uniqids = ipd.dev.UniqueIDs(pymol_chains)
    atoms.chain_id = uniqids(atoms.chain_id, reset=True)

    # ipd.icv(tag)
    with tempfile.TemporaryDirectory() as td:
        td = '/tmp'
        ipd.atom.dump(atoms, f'{td}/{tag}.pdb')

        # with open(f'{td}/{tag}.pdb') as inp:
        # print(inp.read()[:201])

        from pymol import cmd
        cmd.load(f'{td}/{tag}.pdb', name)

@lazy_register('AtomArray')
def regester_atomarray():

    import biotite.structure as bs

    @ipd.viz.pymol_viz.pymol_load.register(bs.AtomArray)
    def pymol_viz_atoms(atoms, name, state, **kw):
        show_atoms_pymol(atoms, name)
