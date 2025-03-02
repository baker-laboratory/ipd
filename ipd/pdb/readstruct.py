from collections.abc import Mapping, Iterable
import io
import ipd

from ipd.lazy_import import lazyimport

bstruc = lazyimport('biotite.structure')
bpdb = lazyimport('biotite.structure.io.pdb')
bpdbx = lazyimport('biotite.structure.io.pdbx')

@ipd.dev.iterize_on_first_param_path
def readatoms(fname, **kw):
    if not ipd.importornone('biotite'):
        raise ImportError('ipd.pdb.readatoms requires biotite')
    with ipd.dev.openfiles(fname, **kw) as file:
        fname = ipd.dev.decompressed_fname(fname)
        if fname.endswith('.pdb'): reader = readatoms_pdb
        elif fname.endswith(('.cif', '.bcif')): reader = readatoms_cif
        struc = reader(fname, file, **kw)
        if isinstance(struc, bstruc.AtomArray):
            struc.__ipd_readatoms_file__ = fname
        elif isinstance(struc, Mapping):
            for v in struc.values():
                v.__ipd_readatoms_file__ = fname
        elif isinstance(struc, Iterable):
            for v in struc:
                v.__ipd_readatoms_file__ = fname
        return struc

def dump(thing, fname):
    if isinstance(thing, (bstruc.AtomArray, bstruc.AtomArrayStack)):
        return dumpatoms(thing, fname)
    assert 0, f'dont know how to dump {type(thing)}'

def dumpatoms(atoms, fname):
    from biotite.structure.io.pdb import PDBFile
    assert fname.endswith('pdb')
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(fname)

def readatoms_cif(fname, file, biounit=None, **kw):
    if biounit:
        (cif, origatoms), nasu = cifread(fname, file, **kw), None
        nasu = len(origatoms)
        if biounit == 'largest':
            assemblies = bpdbx.list_assemblies(cif)
            atoms = []
            for aid in assemblies.keys():
                assembly = bpdbx.get_assembly(cif, aid, model=1)
                if len(assembly) > len(atoms): atoms = assembly
        else:
            atoms = bpdbx.get_assembly(cif, biounit)
        atoms = post_read(atoms, **kw)
        if nasu: atoms = ipd.atom.split(atoms, order=len(atoms) // nasu)
    else:
        cif, atoms = cifread(fname, file, **kw)
    return atoms

def readatoms_pdb(fname, file, **kw):
    pdb, atoms = pdbread(fname, file, **kw)
    return atoms

def pdbread(fname, file=None, **kw):
    if ipd.dev.isbinfile(file):
        file = io.StringIO(file.read().decode())
    pdb = bpdb.PDBFile.read(file or fname)
    atoms = bpdb.get_structure(pdb)
    return pdb, post_read(atoms, **kw)

def cifread(fname, file=None, **kw):
    isbin = fname.endswith('.bcif')
    if ipd.dev.isbinfile(file) and not isbin:
        file = io.StringIO(file.read().decode())
    reader = bpdbx.BinaryCIFFile if isbin else bpdbx.CIFFile
    cif = reader.read(file or fname)
    # pdb = bpdb.PDBFile()
    # pdb.set_structure(cif)
    atoms = bpdbx.get_structure(cif)
    # atoms._spacegroup = pdb.get_space_group()
    return cif, post_read(atoms, **kw)

def cifdump(fname, atoms):
    if not fname.endswith('.bcif'): fname += '.bcif'
    if isinstance(atoms, dict): atoms = atoms.values()
    if not isinstance(atoms, bstruc.AtomArray): atoms = bstruc.concatenate(atoms)
    pdbx = bpdb.PDBFile()
    bstruc.set_structure(pdbx, atoms)
    bcif_file = bpdbx.BinaryCIFFile.from_pdbx_file(pdbx_file)
    bcif_file.write(fname)

def post_read(atoms, chainlist=False, caonly=False, chaindict=False, het=True, **kw):
    if isinstance(atoms, bstruc.AtomArrayStack):
        assert len(atoms) == 1
        atoms = atoms[0]
    if caonly: atoms = atoms[atoms.atom_name == 'CA']
    if not het: atoms = atoms[~atoms.hetero]
    if chaindict: atoms = ipd.atom.chain_dict(atoms)
    if chainlist: atoms = ipd.atom.split(atoms)
    return atoms
