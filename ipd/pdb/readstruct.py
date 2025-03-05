import itertools
import io
from collections.abc import Mapping, Iterable

import numpy as np

import ipd
from ipd.lazy_import import lazyimport

bs = lazyimport('biotite.structure')
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
        if isinstance(struc, bs.AtomArray):
            struc.__ipd_readatoms_file__ = fname
        elif isinstance(struc, Mapping):
            for v in struc.values():
                v.__ipd_readatoms_file__ = fname
        elif isinstance(struc, Iterable):
            for v in struc:
                v.__ipd_readatoms_file__ = fname
        return struc

def dump(thing, fname):
    if isinstance(thing, (bs.AtomArray, bs.AtomArrayStack)):
        return dumpatoms(thing, fname)
    assert 0, f'dont know how to dump {type(thing)}'

def dumpatoms(atoms, fname):
    from biotite.structure.io.pdb import PDBFile
    assert fname.endswith('pdb')
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(fname)

def readatoms_cif(fname, file, assembly=None, **kw):
    if assembly:
        (cif, origatoms), nasu = cifread(fname, file, **kw), None
        nasu = len(origatoms)
        if assembly == 'largest':
            assemblies = bpdbx.list_assemblies(cif)
            atoms = []
            for aid in assemblies.keys():
                assembly = bpdbx.get_assembly(cif, aid, model=1)
                if len(assembly) > len(atoms): atoms = assembly
            # pdb = bpdb.PDBFile()
            # pdb.set_structure(cif)
            # assemblies = bpdb.list_assemblies(pdb)
            # ic(cif.block['pdbx_struct_assembly_gen']['asym_id_list'].as_array(str))
            # ic(cif_asym_id(cif))
            # # ic(cif.block['pdbx_struct_assembly_gen']['oper_expression'].as_array(str))
            # # cif.block["pdbx_struct_oper_list"]
            # xforms = cif_xforms(cif)
            # ic(xforms)
            # # assert 0
            # assembly = max(xforms, key=lambda k: len(xforms[k]))
        else:
            atoms = bpdbx.get_assembly(cif, assembly, model=1)
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
    if not isinstance(atoms, bs.AtomArray): atoms = bs.concatenate(atoms)
    pdbx = bpdb.PDBFile()
    bs.set_structure(pdbx, atoms)
    bcif_file = bpdbx.BinaryCIFFile.from_pdbx_file(pdbx_file)
    bcif_file.write(fname)

def post_read(atoms, chainlist=False, caonly=False, chaindict=False, het=True, **kw):
    if isinstance(atoms, bs.AtomArrayStack):
        assert len(atoms) == 1
        atoms = atoms[0]
    if caonly: atoms = atoms[atoms.atom_name == 'CA']
    if not het: atoms = atoms[~atoms.hetero]
    if chaindict: atoms = ipd.atom.chain_dict(atoms)
    if chainlist: atoms = ipd.atom.split(atoms)
    return atoms

def cif_asym_id(cif):
    chains = cif.block['pdbx_struct_assembly_gen']['asym_id_list'].as_array(str)
    # chains = [c.split(',') for c in chains]
    return list(chains)

def cif_xforms(cif):
    """
    Get transformation operation in terms of rotation matrix and
    translation for each operation ID in ``pdbx_struct_oper_list``.
    """
    struct_oper = cif.block["pdbx_struct_oper_list"]
    transformation_dict = {}
    for index, id in enumerate(struct_oper["id"].as_array(str)):
        rotation_matrix = np.array(
            [[struct_oper[f"matrix[{i}][{j}]"].as_array(float)[index] for j in (1, 2, 3)] for i in (1, 2, 3)])
        translation_vector = np.array([struct_oper[f"vector[{i}]"].as_array(float)[index] for i in (1, 2, 3)])
        transformation_dict[id] = ipd.homog.hconstruct(rotation_matrix, translation_vector)
    return transformation_dict

def cif_opers(cif):
    return [cif_parse_oper(o) for o in cif.block['pdbx_struct_assembly_gen']["oper_expression"].as_array(str)]

def cif_parse_oper(expression):
    """
    Get successive operation steps (IDs) for the given ``oper_expression``.
    Form the cartesian product, if necessary.
    """
    # Split groups by parentheses:
    # use the opening parenthesis as delimiter
    # and just remove the closing parenthesis
    # example: '(X0)(1-10,21-25)' from 1a34
    expressions_per_step = expression.replace(")", "").split("(")
    expressions_per_step = [e for e in expressions_per_step if len(e) > 0]
    # Important: Operations are applied from right to left
    expressions_per_step.reverse()

    operations = []
    for one_step_expr in expressions_per_step:
        one_step_op_ids = []
        for expr in one_step_expr.split(","):
            if "-" in expr:
                # Range of operation IDs, they must be integers
                first, last = expr.split("-")
                one_step_op_ids.extend([str(id) for id in range(int(first), int(last) + 1)])
            else:
                # Single operation ID
                one_step_op_ids.append(expr)
        operations.append(one_step_op_ids)

    # Cartesian product of operations
    return list(itertools.product(*operations))
