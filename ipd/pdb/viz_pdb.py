import random
import tempfile

from ipd.pdb.pdbfile import PDBFile
from ipd.viz.pymol_viz import pymol_load

@pymol_load.register(PDBFile)  # type: ignore
def pymol_viz_pdbfile(
    pdb,
    name,
    state,
    **kw,
):
    tag = str(random.random())[2:]
    # ic(tag)
    with tempfile.TemporaryDirectory() as td:

        pdb.dump_pdb(f'{td}/{tag}.pdb')

        from pymol import cmd  # type: ignore
        cmd.load(f'{td}/{tag}.pdb')
