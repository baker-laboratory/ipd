import functools
import itertools
import io
import os
import typing
import numpy as np

import ipd
from ipd import lazyimport
from ipd import hnumpy as h

if typing.TYPE_CHECKING:
    from biotite.structure import AtomArray
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, BinaryCIFFile

bs = lazyimport('biotite.structure')
bpdb = lazyimport('biotite.structure.io.pdb')
bpdbx = lazyimport('biotite.structure.io.pdbx')

@ipd.dev.iterize_on_first_param_path
@functools.lru_cache
def readatoms(fname, **kw) -> 'AtomArray|list[AtomArray]':
    fname = str(fname)
    if not ipd.importornone('biotite'):
        raise ImportError('ipd.pdb.readatoms requires biotite')
    if not os.path.exists(fname):
        fname = ipd.dev.package_testcif_path(fname)
    with ipd.dev.openfiles(fname, **kw) as file:
        fname = ipd.dev.decompressed_fname(fname)
        if fname.endswith('.pdb'): reader = _readatoms_pdb
        elif fname.endswith(('.cif', '.bcif')): reader = _readatoms_cif
        else: raise ValueError(f'bad filename {fname}')
        atoms = reader(fname, file, **kw)
        ipd.dev.set_metadata(atoms, fname=fname, pdbcode=ipd.Path(fname).stem)
        return atoms

def dump(thing, fname):
    if isinstance(thing, (bs.AtomArray, bs.AtomArrayStack)):
        return dumpatoms(thing, fname)
    assert 0, f'dont know how to dump {type(thing)}'

def dumpatoms(atoms, fname):
    if fname.endswith('pdb'):
        pdb = bpdb.PDBFile()
        pdb.set_structure(atoms)
    elif fname.endswith('.cif'):
        pdb = bpdbx.CIFFile()
        bpdbx.set_structure(pdb, atoms)
    elif fname.endswith('.bcif'):
        pdb = bpdbx.BinaryCIFFile()
        bpdbx.set_structure(pdb, atoms)
    else:
        raise ValueError(f'bad dump filename {fname}')
    pdb.write(fname)

def pdbread(fname, file=None, **kw) -> 'tuple[PDBFile, AtomArray]':
    if ipd.dev.isbinfile(file):
        assert file, 'bad file (fname)'
        file = io.StringIO(file.read().decode())
    pdb = bpdb.PDBFile.read(file or fname)
    atoms = bpdb.get_structure(pdb)
    return pdb, ipd.atom.select(atoms, **kw)

def biotite_cif_file(fname, file=None) -> 'CIFFile|BinaryCIFFile':
    isbin = fname.endswith('.bcif')
    if ipd.dev.isbinfile(file) and not isbin:
        assert file, f'bad file {file}'
        file = io.StringIO(file.read().decode())
    reader = bpdbx.BinaryCIFFile if isbin else bpdbx.CIFFile
    return reader.read(file or fname)

def cifread(fname, file=None, model=0, postproc=True, **kw) -> 'tuple[CIFFile|BinaryCIFFile, AtomArray]':
    cif = biotite_cif_file(fname, file)
    atoms = bpdbx.get_structure(cif)
    if isinstance(atoms, bs.AtomArrayStack): atoms = atoms[model]
    pdbcode = os.path.basename(fname).split('.')[0]
    ipd.dev.set_metadata([cif, atoms], pdbcode=pdbcode, fname=fname)
    if postproc: atoms = ipd.atom.select(atoms, **kw)
    return cif, atoms

def cifdump(fname, atoms):
    if not fname.endswith('.bcif'): fname += '.bcif'
    if isinstance(atoms, dict): atoms = atoms.values()
    if not isinstance(atoms, bs.AtomArray): atoms = bs.concatenate(atoms)
    pdbx = bpdb.PDBFile()
    bs.set_structure(pdbx, atoms)
    bcif_file = bpdbx.BinaryCIFFile.from_pdbx_file(pdbx)
    bcif_file.write(fname)

@ipd.dev.timed
def _readatoms_cif(fname, file, assembly=None, **kw) -> 'AtomArray|list[AtomArray]':
    if assembly is not None:
        return _readatoms_cif_assembly(fname, file, assembly, **kw)
    _, atoms = cifread(fname, file, **kw)
    return atoms

@ipd.dev.timed
def _readatoms_cif_assembly(
    fname,
    file,
    assembly,
    caonly=False,
    het=True,
    **kw,
) -> 'AtomArray|list[AtomArray]':
    cif, asu = cifread(fname, file, caonly=caonly, het=het)
    asminfo = cif_assembly_info(cif)
    if assembly == 'largest':
        best = 0, None
        for ord, asmid in zip(asminfo.assemblies.order, asminfo.assemblies.id):
            if ord > best[0]: best = ord, asmid
        assembly = best[1]
    atoms = bpdbx.get_assembly(cif, assembly, model=1)
    atoms = ipd.atom.select(atoms, caonly=caonly, het=het)
    asmx = _validate_cif_assembly(cif, asminfo, assembly, asu, atoms, **kw)
    atoms = ipd.kwcall(kw, ipd.atom.split, atoms, order=len(asmx._xforms))
    ipd.dev.set_metadata(atoms, assembly_xforms=asmx)
    return atoms

@ipd.dev.timed
def _validate_cif_assembly(cif, asminfo, assembly, asu, atoms, strict=True, **_):
    if assembly in asminfo.assemblies.id:
        iasm = asminfo.assemblies.id.index(assembly)
    else:  # seems sometimes assembly ids don't match annotation... try as numerical index
        iasm = int(assembly) - 1
    _asmid, opers, _asymids, _order = asminfo.assemblies.valwise[iasm]
    xforms = np.array([h.product(*[asminfo.xforms[op] for op in opstep]) for opstep in opers])
    asu = ipd.atom.select(asu, chains=np.unique(atoms.chain_id))
    # ic(asmid, opers, asymids, order)
    # ic(len(asu), len(atoms), asu.coord.shape, atoms.coord[:len(asu):].shape)
    # ic(len(atoms), len(asu), len(atoms) / len(asu))
    err = f'number of atoms {len(atoms)} is not a multiple of the number of ASU atoms {len(asu)}'
    assert len(atoms) % len(asu) == 0, err
    f'number of ASU atoms {len(asu)} times number of assemblies {len(xforms)} does not match number of atoms {len(atoms)}'
    assert len(atoms) // len(asu) == len(xforms), err
    # chainlen = ipd.atom.chainlen(atoms)
    chainsasu, chains = map(ipd.atom.chain_id_ranges, (asu, atoms))
    if len(chains) == 1:
        breaks = ipd.first(chains.values())
        breaks[0] = ipd.first(chainsasu.values())[0]
        assert len(breaks) == 1, f'expected only one chain in assembly {assembly} but got {len(chains)}'
        for _ in range(len(xforms) - 1):
            breaks.append((breaks[-1][0] + len(asu), breaks[-1][1] + len(asu)))
    for c, crng in chainsasu.items():
        for _lb, ub in crng:
            assert 0 < ub <= len(asu), f'bad chain range {crng} for {c} in ASU {len(asu)}'
    for c, crng in chains.items():
        for _lb, ub in crng:
            assert 0 < ub <= len(atoms), f'bad chain range {crng} for {c} in assembly {len(atoms)}'
        # ic(breaks[-1], len(atoms) + 1)
        # assert 0
        # assert breaks[-1][1] == len(atoms) + 1
    # ic(chainsasu)
    # ic(chains)
    # ic(xforms)
    # ic(asminfo)
    asmx = ipd.Bunch(_xforms=xforms, ix=[], asurange=[], symrange=[])
    for ix, _x in enumerate(xforms):
        for c, crngasu, crng in ipd.dev.zipitems(chainsasu, chains):
            num_asu_ranges = len(crngasu)
            # ic(len(xforms), ix, c, len(crng), num_asu_ranges)
            assert len(xforms) == len(
                crng
            ) / num_asu_ranges, f'number of assemblies {len(xforms)} does not match number of chains {len(crng) / num_asu_ranges} for {c}'
            for iasurange in range(num_asu_ranges):
                (lb1, ub1), (lb2, ub2) = crngasu[iasurange], crng[ix*num_asu_ranges + iasurange]
                # ic(ix, c, iasurange)
                assert ub1 - lb1 == ub2 - lb2, f'number of atoms in ASU {ub1 - lb1} does not match number of atoms in assembly {ub2 - lb2} for {c}'
                asmx.ix.append(ix)
                asmx.asurange.append((lb1, ub1))
                asmx.symrange.append((lb2, ub2))
                if strict:
                    orig = h.xform(xforms[ix], asu.coord[lb1:ub1])
                    new = atoms.coord[lb2:ub2]
                    # ic(orig.shape, new.shape)
                    # ipd.showme(orig, new)
                    close = np.allclose(orig, new, atol=1e-3)
                    if not close:
                        pdbcode = ipd.dev.get_metadata(cif).get('pdbcode')
                        ipd.dev.WARNME(
                            f'{pdbcode} biounit {assembly} {ix} {c} failed coordinate symmetry check',
                            verbose=False)
                        assert close, 'failde coordinate symmetry check'
    asmx = asmx.mapwise(np.array)
    asmx._chainasu, asmx._chains = chainsasu, chains
    # assert np.allclose(asu.coord, atoms.coord[:len(asu)], atol=1e-3)
    # ic(ipd.bunch.zip(asymchainstart, asymchainlen, chainstart, chainlen, order='val'))
    return asmx

def _readatoms_pdb(fname, file, **kw) -> 'AtomArray':
    _pdb, atoms = pdbread(fname, file, **kw)
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
            assert op in xforms, f'bad oper {op} in {op_expr} {xforms.keys()}'
        assemblies.mapwise.append(id=str(id), oper=op_expr, asymids=asymids, order=int(order))
    # assemblies.id = np.array(assemblies.id)
    return ipd.Bunch(assemblies=assemblies, xforms=xforms)
