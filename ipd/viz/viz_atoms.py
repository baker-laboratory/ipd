import random
import tempfile

import ipd

@ipd.viz.pymol_viz.lazy_register
def regester_atomarray():

    bs = ipd.importornone('biotite.structure')

    @ipd.viz.pymol_viz.pymol_load.register(bs.AtomArray)
    def pymol_viz_atoms(
        atoms,
        name,
        state,
        **kw,
    ):
        tag = str(random.random())[2:]
        # ic(tag)
        with tempfile.TemporaryDirectory() as td:

            ipd.atom.dump(atoms, f'{td}/{tag}.pdb')

            from pymol import cmd
            cmd.load(f'{td}/{tag}.pdb')
