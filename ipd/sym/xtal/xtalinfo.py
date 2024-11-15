import numpy as np

import ipd
from ipd.homog import *
from ipd.sym.xtal.SymElem import SymElem

_xtal_asucens = {
    "P 4 3 2": np.array([0.1, 0.2, 0.3, 1]),
    "P 4 3 2 43": np.array([0.1, 0.2, 0.3, 1]),
    "P 4 3 2 44": np.array([0.1, 0.2, 0.3, 1]),
    "I 4 3 2": np.array([0.28, 0.17, 0.08, 1]),
    # 'I 4 3 2 432': np.array([0.28, 0.17, 0.08, 1]),
    "I 4 3 2 432": np.array([-0.0686, 0.037, 0.133, 1]),
    "F 4 3 2": np.array([0.769, 0.077, 0.385, 1.0]),
    # 'F 4 3 2': np.array([0.714, 0.071, 0.357, 1.0]),
    "L6_32": np.array([0.2886751345948129, 0, 0, 1]),
    "L4_42": np.array([0.31, 0, 0, 1]),
    "L4_44": np.array([0.25, 0, 0, 1]),
    "L3_33": np.array([0.25, 0, 0, 1]),
    # 'I 21 3': np.array([0.615, 0.385, 0.615, 1.0]),
    # 'I 21 3': np.array([0.577, 0.385, 0.615, 1.0]),
    "I 21 3": np.array([0.357, 0.357, 0.643, 1.0]),
    "P 21 3": np.array([0.429, 0.214, 0.5, 1.0]),
    #
    "I4132_322": np.array([-0.08385417, 0.0421875, 0.14791667, 1]),
}

def all_xtal_names():
    if _xtal_info_dict is None:
        _populate__xtal_info_dict()
    allxtals = [k for k in _xtal_info_dict if not k.startswith("DEBUG")]  # type: ignore
    return allxtals

def _populate__xtal_info_dict():
    global _xtal_info_dict
    A = np.array
    ##################################################################################
    ######## IF YOU CHANGE THESE, REMOVE CACHE FILES OR DISABLE FRAME CACHING ########
    ######## IF YOU CHANGE THESE, REMOVE CACHE FILES OR DISABLE FRAME CACHING ########
    ######## IF YOU CHANGE THESE, REMOVE CACHE FILES OR DISABLE FRAME CACHING ########
    ######## IF YOU CHANGE THESE, REMOVE CACHE FILES OR DISABLE FRAME CACHING ########
    ##################################################################################
    # yapf: disable
    _xtal_info_dict = {
      # 'P 4 3 2'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    # C4 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 1 ]) / 2 ),
      #    C4 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C4 ( axis= [ 0,  1,  0 ] , cen= A([ 1, 0, 1 ]) / 2 ),
      # ]),
      # 'P 4 3 2 443'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    C4 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 1 ]) / 2 ),
      #    C4 ( axis= [ 0,  1,  0 ] , cen= A([ 1, 0, 1 ]) / 2 ),
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      # ]),
      # 'P 4 3 2 43'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    C4 ( axis= [ 0,  1,  0 ] , cen= A([ 1, 0, 1 ]) / 2 ),
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      # ]),
      # 'P 4 3 2 43'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    C4 ( axis= [ 0,  1,  0 ] , cen= A([ 1, 0, 1 ]) / 2 ),
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      # ]),
      # 'P 4 3 2 322'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  1,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 0 ]) / 2 ),
      # ]),
      # 'P 4 3 2 422'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    C4 ( axis= [ 0,  1,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  1,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 0 ]) / 2 ),
      # ]),
      # 'P 4 3 2 432'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    C4 ( axis= [ 0,  1,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 0 ]) / 2 ),
      # ]),
      # 'P 4 3 2 4322'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    C4 ( axis= [ 0,  1,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  1,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 0 ]) / 2 ),
      # ]),
      # 'P 4 3 2 432D2'   : ipd.dev.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
      #    C4 ( axis= [ 0,  1,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  1,  0 ] , cen= A([ 0, 0, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 0 ]) / 2 ),
      #    C2 ( axis= [ 1,  0,  1 ] , cen= A([ 0, 1, 0 ]) / 2 ),
      # ]),
      # 'F 4 3 2'   : ipd.dev.Bunch( nsub=96 , spacegroup='F 4 3 2', symelems=[
      #    C4 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 0, 1 ]) / 2 ),
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 2,-1,-1 ]) / 6 ),
      #    # C4 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 1 ]) / 2 ),
      #    # C3 ( axis= [ 1,  1,  1 ] , cen= A([ 4, 1, 1 ]) / 6 ),
      #    # C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 6 ),
      # ]),
      # 'I 4 3 2 432'   : ipd.dev.Bunch( nsub=48 , spacegroup='I 4 3 2', symelems=[
      #    C4 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
      #    C3 ( axis= [ 1,  1, -1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
      #    C2 ( axis= [ 0,  1,  1 ] , cen= A([ 1, 1,-1 ]) / 4 ),
      #    # C4 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 1, 0 ]) / 2 ),
      #    # C2 ( axis= [ 0,  1,  1 ] , cen= A([ 1, 1, 1 ]) / 2 ),
      # ]),

      # 'P 2 3'    : ipd.dev.Bunch( nsub=12 , spacegroup='P 2 3', symelems=[
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 1, 1, 1 ]) / 2, label='C3_111_000' ),
      #    C2 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 1, 0 ]) / 2, label='C2_001_000' ),
      #    # C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 0 ]) / 2, label='C2_100_010' ),
      # ]),
      # 'P 21 3'   : ipd.dev.Bunch( nsub=12 , spacegroup='P 21 3', symelems=[
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2, vizcol=(0.0, 1.0, 1.0), label='A' ),
      #    C3 ( axis= [ 1,  1, -1 ] , cen= A([ 1, 0, 1 ]) / 2, vizcol=(0.3, 1, 0.7), label='B' ),
      # ]),
      # 'I 21 3'   : ipd.dev.Bunch( nsub=24 , spacegroup='I 21 3', symelems=[
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
      #    C2 ( axis= [ 0,  0,  1 ] , cen= A([ 2, 1, 0 ]) / 4 ),
      # ]),
      # 'P 41 3 2'  : ipd.dev.Bunch( nsub=24 , spacegroup='P 41 3 2', symelems=[
      #    C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
      #    C2 ( axis= [ 1,  0,  1 ] , cen= A([ 2, 1, 0 ]) / 8 ),
      # ]),
      # 'I 41 3 2' : ipd.dev.Bunch( nsub=48, spacegroup='I 41 3 2', symelems=[
      # D3 ( axis= [ 1,  1,  1 ] , axis2= [ 1, -1,  0 ] , cen= A([ 1, 1, 1 ]) / 8, label='D3_111_1m0_111_8' , vizcol=(0, 1, 0)),
      # D2 ( axis= [ 1,  0,  0 ] , axis2= [ 0, -1,  1 ] , cen= A([ 1, 0, 2 ]) / 8, label='D2_100_0m1_102_8' , vizcol=(0, 1, 1)),
      # D3 ( axis= [ 1,  1,  1 ] , axis2= [ 1, -1,  0 ] , cen= A([-1,-1,-1 ]) / 8, label='D3_111_1m0_mmm_8' , vizcol=(1, 0, 0)),
      # D2 ( axis= [ 1,  0,  0 ] , axis2= [ 0, -1,  1 ] , cen= A([-1, 0,-2 ]) / 8, label='D2_100_0m1_m12m_8', vizcol=(1, 1, 0)),
      # ]),
      # #'I 41 3 2' : ipd.dev.Bunch( nsub=48, spacegroup='I 41 3 2', symelems=[
      # #   C3 ( axis= [ 1,  1,  1 ] , cen= A([ 1, 1, 1 ]) / 8, label='D3_111_1m0_111_8' , vizcol=(0, 1, 0)),
      # #   C2 ( axis= [ 1, -1,  0 ] , cen= A([ 1, 1, 1 ]) / 8, label='D3_111_1m0_111_8' , vizcol=(0, 1, 0)),
      # #   C2 ( axis= [ 1,  0,  0 ], cen= A([ 1, 0, 2 ]) / 8, label='D2_100_0m1_102_8' , vizcol=(0, 1, 1)),
      # #   C2 ( axis= [ 0, -1,  1 ] , cen= A([ 1, 0, 2 ]) / 8, label='D2_100_0m1_102_8' , vizcol=(0, 1, 1)),
      # #   C3 ( axis= [ 1,  1,  1 ] , cen= A([-1,-1,-1 ]) / 8, label='D3_111_1m0_mmm_8' , vizcol=(1, 0, 0)),
      # #   C2 ( axis= [ 1, -1,  0 ] , cen= A([-1,-1,-1 ]) / 8, label='D3_111_1m0_mmm_8' , vizcol=(1, 0, 0)),
      # #   C2 ( axis= [ 1,  0,  0 ] , cen= A([-1, 0,-2 ]) / 8, label='D2_100_0m1_m12m_8', vizcol=(1, 1, 0)),
      # #   C2 ( axis= [ 0, -1,  1 ] , cen= A([-1, 0,-2 ]) / 8, label='D2_100_0m1_m12m_8', vizcol=(1, 1, 0)),
      # #]),
      'I4132_322' : ipd.dev.Bunch( nsub=48, spacegroup='I 41 3 2', symelems=[
         # C3 ( axis= [ 1,  1,  1 ] , cen= A([ 2, 2, 2 ]) / 8, label='C3_111_1m0_111_8' , vizcol=(1, 0, 0)),
         # C2 ( axis= [ 1,  0,  0 ] , cen= A([ 3, 0, 2 ]) / 8, label='D2_100_0m1_102_8' , vizcol=(0, 1, 0)),
         # C2 ( axis= [ 1, -1,  0 ] , cen= A([-2.7, 0.7,-1 ]) / 8, label='D3_111_1m0_mmm_8' , vizcol=(0, 0, 1)),
         C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 8, label='C3_111_1m0_111_8' , vizcol=(1, 0, 0)),
         C2 ( axis= [ 1,  0,  0 ] , cen= A([-1, 0, 2 ]) / 8, label='D2_100_0m1_102_8' , vizcol=(0, 1, 0)),
         C2 ( axis= [ 1,  1,  0 ] , cen= [-0.1625,  0.0875,  0.125 ], label='D3_111_1m0_mmm_8' , vizcol=(0, 0, 1)),
      ]),
      'L6_32'   : ipd.dev.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C3 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C2 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
      ]),
      'L6M_322' : ipd.dev.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C3 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C2 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
         C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.5, 1, 0.8) ),
      ]),
      'L4_44'   : ipd.dev.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C4 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C4 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
      ]),
      'L4_42'   : ipd.dev.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C4 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C2 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
      ]),
      'L3_33'   : ipd.dev.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C3 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C3 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
      ]),


    }
    # yapf: enable
    from ipd.sym.xtal.spacegroup_deriveddata import (
        sg_cheshire_dict,
        sg_frames_dict,
        sg_symelem_dict,
        sg_symelem_frame444_compids_dict,
        sg_symelem_frame444_opcompids_dict,
        sg_symelem_frame444_opids_dict,
    )
    for k in sg_frames_dict:
        if k not in sg_symelem_dict: continue
        _xtal_info_dict[k] = ipd.dev.Bunch(
            spacegroup=k,
            nsub=len(sg_frames_dict[k]),  # type: ignore
            dimension=3,
            symelems=sg_symelem_dict[k],  # type: ignore
            frames=sg_frames_dict[k],  # type: ignore
            cheshire=sg_cheshire_dict[k],  # type: ignore
            opids4=sg_symelem_frame444_opids_dict[k],  # type: ignore
            compids4=sg_symelem_frame444_compids_dict[k],  # type: ignore
            opcompids4=sg_symelem_frame444_opcompids_dict[k],  # type: ignore
            _strict=True,  # type: ignore
        )

def C2(**kw):
    return SymElem(nfold=2, **kw)

def C3(**kw):
    return SymElem(nfold=3, **kw)

def C4(**kw):
    return SymElem(nfold=4, **kw)

def C6(**kw):
    return SymElem(nfold=6, **kw)

def D2(**kw):
    return SymElem(nfold=2, **kw)

def D3(**kw):
    return SymElem(nfold=3, **kw)

def D4(**kw):
    return SymElem(nfold=4, **kw)

def D6(**kw):
    return SymElem(nfold=6, **kw)

_xtal_info_dict = None

def is_known_xtal(name):
    try:
        xtalinfo(name)
        return True
    except (KeyError, ValueError):
        return False

def xtalinfo(name):
    if _xtal_info_dict is None:
        _populate__xtal_info_dict()

    name = name.upper().strip()

    if name not in _xtal_info_dict:
        alternate_names = {
            "P432": "P 4 3 2",
            "P432_43": "P 4 3 2 43",
            "F432": "F 4 3 2",
            "I432": "I 4 3 2 432",
            "I432_432": "I 4 3 2 432",
            "I 4 3 2": "I 4 3 2 432",
            "I4132": "I 41 3 2",
            "P4132": "P 41 3 2",
            "P213": "P 21 3",
            "P213_33": "P 21 3",
            "I213_32": "I 21 3",
            "I213": "I 21 3",
            "L6M322": "L6M_322",
            "L632": "L6_32",
            "P6_32": "L6_32",
            "P4_42": "L4_42",
            "P4_44": "L4_44",
            "P3_33": "L3_33",
            "P23": "P 2 3",
        }

        if name in alternate_names:
            name = alternate_names[name]
        else:
            raise ValueError(f'unknown xtal {name}')
    if name not in _xtal_info_dict:  # type: ignore
        name = name.replace("_", " ")
    # ic(name)
    info = _xtal_info_dict[name]  # type: ignore
    return name, info

"""
P212121
P43212
P3121
C2221,P2
P3221,P6122 
P6522
P41212
I4,P61,R3 
P31,P41,P43
P32,P6,P63,P65
I41,P62,P64
P213,123,1213,P4132,P4232,P4332P432,1432,14132F432,F4132 
"""
