"""
Module: ipd.sym.sym
===================

This module handles symmetry-related operations within the IPD library.
It defines functions and classes for creating and manipulating symmetry bodies (SymBody),
which are extended representations of a Body that incorporate the notion of symmetry.
These are particularly useful when working with large symmetrical structures where
efficient spatial queries are critical.

Key features:
  - Creation of SymBody objects from file data.
  - Application of homogeneous transformations (using hgeom) to symmetrical structures.
  - Fast clash and contact checks using SphereBVH_double.

Usage Examples:
    >>> from ipd import sym, hgeom as h, atom
    >>> # Create a SymBody using a built-in helper function and a pdb code
    >>> sb = sym.symbody_from_file("1A0J")
    >>> # Apply a random transformation
    >>> T = h.rand()
    >>> sb_transformed = sb.apply_transform(T)

    >>> # Clash detection between two symmetry bodies
    >>> sb2 = sym.symbody_from_file("1bfr")
    >>> result = sb.clash(sb2)
    >>> isinstance(result, bool)
    True

    >>> # Checking contacts with an AtomArray
    >>> aa = atom.load("1A0J")
    >>> contacts = sb.contact_check(aa)
    >>> isinstance(contacts, list)
    True

Additional Examples:
    >>> # Compose a translation and a rotation and apply to a SymBody
    >>> T_trans = h.trans([1, 2, 3])
    >>> T_rot = h.rot([0, 1, 0], 45, [0, 0, 0])
    >>> T_combined = h.xform(T_trans, T_rot)
    >>> sb_new = sb.apply_transform(T_combined)

    >>> # Performance demonstration (conceptual timing check)
    >>> import time
    >>> start = time.time()
    >>> _ = sb.clash(sb2)
    >>> elapsed = time.time() - start
    >>> elapsed < 1.0  # Expect fast computation
    True

.. note::
    Additional tests and examples can be found in ipd/tests/test_sym_detect.py.
"""
import os
import traceback

import numpy as np

import ipd
from ipd.tools.ipdtool import IPDTool
from ipd.tools.tool_util import RunGroupArg

custom_tol = dict(default=1e-1,
                  angle=0.04,
                  helical_shift=4,
                  isect=6,
                  dot_norm=0.07,
                  misc_lineuniq=0.2,
                  rms_fit=3,
                  nfold=0.2)
tol = ipd.dev.Tolerances(**(ipd.sym.symdetect_default_tolerances | custom_tol))

class SymTool(IPDTool):
    pass

class BuildTool(SymTool):

    def from_components(self, comps: list[str]):
        print(comps)
        assert 0, 'not implemented'

class TestTool(SymTool):

    def detect(self, fnames: list[str], rungroup: RunGroupArg):
        for i, fname in ipd.tools.enumerate_inputs(fnames, '*.bcif.gz', rungroup):
            pdbcode = fname.stem.split('.')[0]
            _sym_check_file(pdbcode, fname, tol)

    def assembly(self, fnames: list[str], rungroup: RunGroupArg):
        for i, fname in ipd.tools.enumerate_inputs(fnames, '*.bcif.gz', rungroup):
            pdbcode = fname.stem.split('.')[0]
            atoms = ipd.body.assembly(fname, assembly='largest', het=False)

    def readstruct(self, fnames: list[str], rungroup: RunGroupArg):
        for i, fname in ipd.tools.enumerate_inputs(fnames, '*.bcif.gz', rungroup):
            pdbcode = fname.stem.split('.')[0]
            size = os.path.getsize(fname) / 1024
            try:
                atoms = ipd.pdb.readatoms(fname, assembly='largest', strict=True)
                print(
                    'readstruct OK', pdbcode,
                    [(int(np.sum(a.atom_name == 'CA')), int(np.sum(a.atom_name == 'P')), int(np.sum(a.hetero)))
                     for a in atoms])
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'readstruct FAIL {pdbcode} {size:9.3f}K {e}', flush=True)

def _sym_check_file(pdbcode, fname, tol):
    symanno = ipd.pdb.sym_annotation(pdbcode)
    for id, symid in zip(symanno.id, symanno.sym):
        if symid == 'C1': continue
        print(f'symdetect {pdbcode} {id:2} {symid:4}', flush=True, end=' ')
        try:
            atoms = ipd.atom.get(pdbcode, assembly=id, het=False, fname=fname.parent)
            sinfo = ipd.sym.detect(atoms, tol=tol)
            if isinstance(sinfo, list):
                print([si.symid for si in sinfo], end=' ')
                sinfo = sinfo[0]
            infer_t = sinfo.pseudo_order // sinfo.order
            if symid != sinfo.symid:
                if not any([
                        symid == 'C2' and sinfo.is_dihedral,
                        symid == 'C2' and sinfo.symid in 'TIO',
                        symid == 'C3' and sinfo.symid in 'TIO',
                        symid == 'C4' and sinfo.symid in 'O',
                        symid == 'C5' and sinfo.symid in 'I',
                ]):
                    print(f'mispredict {sinfo.symid}', end=' ', flush=True)
                else:
                    print('OK', end=' ')
            elif sinfo.t_number != infer_t:
                print(f'bad T {infer_t} {syminfo.t_number}', end=' ', flush=True)
            else:
                print('OK', end='')
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]  # Get the last traceback frame
            print(f"{type(e).__name__}: {e} {tb.filename.split('/')[-1]}:{tb.lineno})", end='')
        print(flush=True)
