"""
This subpackage provides a comprehensive toolkit for atom-level operations within the IPD framework. It aggregates functionalities for processing and analyzing atomic structures, including utilities, alignment components, body representations, and assembly manipulations.

Modules:
    - atom_utils:
        Provides helper functions and utilities for common atom-level operations such as chain splitting, coordinate transformations, and sequence alignment support.
    - components:
        Contains classes and functions for aligning atomic structures based on sequence alignment and RMSD
        fitting, and for managing the resulting alignment data.
    - body:
        Defines building blocks for constructing higher-level atomic body representations and related operations.
    - assembly:
        Offers tools for handling and manipulating assemblies of atomic structures, including merging, splitting, and transforming multiple chains.

Usage Example:
    >>> atoms = ipd.atom.load('1dxh', assembly='largest', het=False, chainlist=True)
    >>> components = ipd.atom.find_components_by_seqaln_rmsfit(atoms)
    >>> print(components)
    Components:
      atoms: [2669]
      frames: [(12, 4, 4)]
      seqmatch: [array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])]
      rmsd: [array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]
      idx: [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])]
      source_: <class 'list'>

Dependencies:
    - numpy
    - biotite (accessed via lazy import)
"""

from ipd.atom.atom_utils import *
from ipd.atom.components import *
from ipd.atom.body import *
from ipd.atom.assembly import *
