from collections.abc import Mapping, Iterable
import functools
import itertools
import io
import os

import numpy as np

import ipd
from ipd.lazy_import import lazyimport

h = ipd.hnumpy

bs = lazyimport('biotite.structure')
bpdb = lazyimport('biotite.structure.io.pdb')
bpdbx = lazyimport('biotite.structure.io.pdbx')

@ipd.dev.iterize_on_first_param_path
@functools.lru_cache
def readatoms(fname, **kw) -> 'Atoms':
    if not ipd.importornone('biotite'):
        raise ImportError('ipd.pdb.readatoms requires biotite')
    if not os.path.exists(fname):
        fname = ipd.dev.package_testcif_path(fname)
    with ipd.dev.openfiles(fname, **kw) as file:
        fname = ipd.dev.decompressed_fname(fname)
        if fname.endswith('.pdb'): reader = _readatoms_pdb
        elif fname.endswith(('.cif', '.bcif')): reader = _readatoms_cif
        atoms = reader(fname, file, **kw)
        add_sourcefile_tag(atoms, fname)
        return atoms

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

def pdbread(fname, file=None, **kw) -> 'tuple[Pdb, Atoms]':
    if ipd.dev.isbinfile(file):
        file = io.StringIO(file.read().decode())
    pdb = bpdb.PDBFile.read(file or fname)
    atoms = bpdb.get_structure(pdb)
    return pdb, ipd.atom.select(atoms, **kw)

def biotite_cif_file(fname, file=None) -> 'Cif':
    isbin = fname.endswith('.bcif')
    if ipd.dev.isbinfile(file) and not isbin:
        file = io.StringIO(file.read().decode())
    reader = bpdbx.BinaryCIFFile if isbin else bpdbx.CIFFile
    return reader.read(file or fname)

def cifread(fname, file=None, postproc=True, **kw) -> 'tuple[Cif, Atoms]':
    # pdb = bpdb.PDBFile()
    # pdb.set_structure(cif)
    cif = biotite_cif_file(fname, file)
    atoms = bpdbx.get_structure(cif)
    # atoms._spacegroup = pdb.get_space_group()
    if postproc: atoms = ipd.atom.select(atoms, **kw)
    return cif, atoms

def cifdump(fname, atoms):
    if not fname.endswith('.bcif'): fname += '.bcif'
    if isinstance(atoms, dict): atoms = atoms.values()
    if not isinstance(atoms, bs.AtomArray): atoms = bs.concatenate(atoms)
    pdbx = bpdb.PDBFile()
    bs.set_structure(pdbx, atoms)
    bcif_file = bpdbx.BinaryCIFFile.from_pdbx_file(pdbx_file)
    bcif_file.write(fname)

def _readatoms_cif(fname, file, assembly=None, **kw) -> 'Atoms':
    if assembly is not None:
        return _readatoms_cif_assembly(fname, file, assembly, **kw)
    cif, atoms = cifread(fname, file, **kw)
    return atoms

def _readatoms_cif_assembly(fname, file, assembly, caonly=False, het=True, **kw) -> 'tuple[Cif, Atoms]':
    cif, asu = cifread(fname, file, caonly=caonly, het=het)
    asminfo = cif_assembly_info(cif)
    if assembly == 'largest':
        _nchain, assembly = max(zip(asminfo.assemblies.order, asminfo.assemblies.id))
    atoms = bpdbx.get_assembly(cif, assembly, model=1)
    atoms = ipd.atom.select(atoms, caonly=caonly, het=het)
    xforms = _validate_cif_assembly(cif, asminfo, assembly, asu, atoms)
    atoms = ipd.atom.split(atoms, order=len(xforms))
    return atoms

def _validate_cif_assembly(cif, asminfo, assembly, asu, atoms):
    i = asminfo.assemblies.id.index(assembly)
    asmid, opers, asymids, order = asminfo.assemblies.valwise[i]
    xforms = np.array([h.product(*[asminfo.xforms[op] for op in opstep]) for opstep in opers])
    asu = ipd.atom.select(asu, chains=np.unique(atoms.chain_id))
    # ic(asmid, opers, asymids, order)
    # ic(len(asu), len(atoms), asu.coord.shape, atoms.coord[:len(asu):].shape)
    # ic(len(atoms), len(asu), len(atoms) / len(asu))
    assert len(atoms) % len(asu) == 0
    assert len(atoms) // len(asu) == len(xforms)
    # chainlen = ipd.atom.chainlen(atoms)
    chainsasu, chains = map(ipd.atom.chain_ranges, (asu, atoms))
    # ic(chainsasu)
    # ic(chains)

    for ix, x in enumerate(xforms):
        for c, crngasu, crng in ipd.bunch.zipitems(chainsasu, chains):
            num_asu_ranges = len(crngasu)
            assert len(xforms) == len(crng) / num_asu_ranges
            for iasurange in range(num_asu_ranges):
                (lb1, ub1), (lb2, ub2) = crngasu[iasurange], crng[ix*num_asu_ranges + iasurange]
                orig = h.xform(xforms[ix], asu.coord[lb1:ub1])
                new = atoms.coord[lb2:ub2]
                # ipd.showme(orig, new)
                assert np.allclose(orig, new, atol=1e-3)
    # assert np.allclose(asu.coord, atoms.coord[:len(asu)], atol=1e-3)
    # ic(ipd.bunch.zip(asymchainstart, asymchainlen, chainstart, chainlen, order='val'))
    return xforms

def _readatoms_pdb(fname, file, **kw) -> 'Atoms':
    pdb, atoms = pdbread(fname, file, **kw)
    return atoms

def _cif_xforms(cif):
    """
    Get transformation operation in terms of rotation matrix and
    translation for each operation ID in ``pdbx_struct_oper_list``.
    """
    struct_oper = cif.block["pdbx_struct_oper_list"]
    xforms = {}
    for index, id in enumerate(struct_oper["id"].as_array(str)):
        rot = np.array([[struct_oper[f"matrix[{i}][{j}]"].as_array(float)[index] for j in (1, 2, 3)]
                        for i in (1, 2, 3)])
        trans = np.array([struct_oper[f"vector[{i}]"].as_array(float)[index] for i in (1, 2, 3)])
        xforms[str(id)] = ipd.homog.hconstruct(rot, trans)
    return xforms

def _cif_opers(cif):
    return [_cif_parse_oper(o) for o in cif.block['pdbx_struct_assembly_gen']["oper_expression"].as_array(str)]

def _cif_parse_oper(expression):
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

def cif_assembly_info(cif):
    block = cif.block
    try:
        asmbl = block["pdbx_struct_assembly"]
        asmblgen = block["pdbx_struct_assembly_gen"]
    except KeyError:
        raise ValueError("File has no 'pdbx_struct_assembly_gen' category")
    # try:
    # struct_oper_category = block["pdbx_struct_oper_list"]
    # except KeyError:
    # raise ValueError("File has no 'pdbx_struct_oper_list' category")
    # opers = _cif_opers(cif)
    ids = asmblgen['assembly_id'].as_array(str)
    assert len(set(ids)) == len(ids)
    assembly_ids = asmblgen["assembly_id"].as_array(str)
    xforms = _cif_xforms(cif)
    assemblies = ipd.Bunch(id=[], oper=[], asymids=[], order=[])
    for id, op_expr, asym_id_expr, order in zip(
            asmblgen["assembly_id"].as_array(str),
            asmblgen["oper_expression"].as_array(str),
            asmblgen["asym_id_list"].as_array(str),
            asmbl['oligomeric_count'].as_array(str),
    ):
        _cif_parse_oper(op_expr)
        asymids = asym_id_expr.split(',')
        op_expr = _cif_parse_oper(op_expr)
        # ic([f'{k}{v.shape}' for k,v in xforms.items()])
        for op in ipd.dev.addreduce(op_expr):
            assert op in xforms
        assemblies.mapwise.append(id=str(id), oper=op_expr, asymids=asymids, order=int(order))
    # assemblies.id = np.array(assemblies.id)
    return ipd.Bunch(assemblies=assemblies, xforms=xforms)

def add_sourcefile_tag(struc, fname):
    if isinstance(struc, bs.AtomArray): struc.__ipd_source_file__ = fname
    elif isinstance(struc, Mapping): map(add_sourcefile_tag, struc.values())
    elif isinstance(struc, Iterable): map(add_sourcefile_tag, struc)
    else: raise TypeError(f'cant add file tag to {type(struc)}')
