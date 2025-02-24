import io
import numpy as np

import ipd

from ipd.lazy_import import lazyimport

bstruc = lazyimport('biotite.structure')
bpdb = lazyimport('biotite.structure.io.pdb')
bpdbx = lazyimport('biotite.structure.io.pdbx')

@ipd.dev.iterize_on_first_param_path
def readatoms(fname, **kw):
    with ipd.dev.openfiles(fname, **kw) as file:
        fname = ipd.dev.decompressed_fname(fname)
        if fname.endswith('.pdb'): reader = readatoms_pdb
        elif fname.endswith(('.cif', '.bcif')): reader = readatoms_cif
        return reader(fname, file, **kw)

def readatoms_cif(*a, **kw):
    cif, atom = cifread(*a, **kw)
    return atom

def readatoms_pdb(*a, **kw):
    pdb, atom = pdbread(*a, **kw)
    return atom

def pdbread(fname, file=None, **kw):
    if ipd.dev.isbinfile(file):
        file = io.StringIO(file.read().decode())
    pdb = bpdb.PDBFile.read(file or fname)
    atom = bpdb.get_structure(pdb)
    return pdb, post_read(atom, **kw)

def cifread(fname, file=None, **kw):
    isbin = fname.endswith('.bcif')
    if ipd.dev.isbinfile(file) and not isbin:
        file = io.StringIO(file.read().decode())
    reader = bpdbx.BinaryCIFFile if isbin else bpdbx.CIFFile
    cif = reader.read(file or fname)
    atom = bpdbx.get_structure(cif)
    return cif, post_read(atom, **kw)

def cifdump(fname, atom):
    if not fname.endswith('.bcif'): fname += '.bcif'
    if isinstance(atom, dict): atom = atom.values()
    if not isinstance(atom, bstruc.AtomArray): atom = bstruc.concatenate(atom)
    pdbx = bpdb.PDBFile()
    bstruc.set_structure(pdbx, atom)
    bcif_file = bpdbx.BinaryCIFFile.from_pdbx_file(pdbx_file)
    bcif_file.write(fname)

def post_read(atoms, bychain=False, caonly=False, **kw):
    assert len(atoms) == 1
    atoms = atoms[0]
    if caonly: atoms = atoms[atoms.atom_name == 'CA']
    if bychain: atoms = group_by_chain(atoms)
    return atoms

def group_by_chain(atom_array):
    """Group an AtomArray by chain_id and return a dictionary."""
    chain_ids = np.unique(atom_array.chain_id)  # Get unique chain IDs
    chain_groups = {chain: atom_array[atom_array.chain_id == chain] for chain in chain_ids}
    return chain_groups
